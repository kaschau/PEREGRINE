import numpy as np

from .restart import restart
from .solverBlock import solverBlock
from ..bcs import getBc
from ..integrators import getIntegrator
from ..jit import Jit
from ..kernel import BoundKernel, NullKernel
from ..mixture import Mixture
from ..mpiComm import Communicator
from ..table import Table
from .thtrdat import thtrdat


class pgConfigError(Exception):
    def __init__(self, setting, option, altMessage=""):
        message = f"Unknown PEREGRINE config {setting} option: {option}. "
        super().__init__(message + altMessage)


class solver(restart):
    """A list of peregrinepy.multiBlock.solver.
    Inherits from peregrinepy.multiBlock.restart"""

    hasConservatives = True

    def _newBlock(self, nblki):
        return solverBlock(
            nblki, self.speciesNames, self.ng, self.config, self, len(self)
        )

    def progress(self, n, message):
        """A running case reports through its own machinery, not a bar."""

    def __init__(self, config, nblks=None, myblocks=None):
        """A runnable case: the blocks, the mixture and its species data, the
        time integrator, and every kernel the config asks for, compiled and
        bound. :myblocks: are the block numbers this rank is responsible for,
        which is also how many blocks it holds."""
        if myblocks is not None:
            nblks = len(myblocks)

        self.config = config
        self.mixture = Mixture(config["mcPhysics"], root=config["io"]["inputDir"])
        self.ng = solverBlock.haloDepth(config)

        # every block's records, which each block fills in as it allocates
        self.table = Table(nblks, self.ng)
        spNames = self.mixture.speciesNames
        super().__init__(
            nblks,
            spNames,
            [solverBlock(i, spNames, self.ng, config, self, i) for i in range(nblks)],
        )

        # in parallel the blocks are numbered by the partition, not by order
        if myblocks is not None:
            for blk, nblki in zip(self, myblocks):
                blk.nblki = blk.baseNblki = nblki

        # time integrator time
        self.titme = 0.0
        self.thtrdat = thtrdat(self.mixture)

        # the kernels compiled for this case, by the name they are called by
        self.jit = Jit(self.mixture.ns, self.ng)
        self.kernels = {}

        # Result output
        self.resultsWriter = None
        # Halo exchange, once the blocks know their neighbors
        self.communicator = None
        # the boundary conditions' kernels by hook, and their faces' records
        self.bcHooks = {}
        self._faceTables = {}

        # what consistify and RHS call, from what the config picks
        self.phiComm = False
        self.setConsistify()
        self.setRHS()
        for name in ("allFinite", "CFLmax", "residual"):
            setattr(self, name, self.kernel(f"utils/{name}.cpp"))
        # how the case steps in time
        self.integrator = getIntegrator(config["timeIntegration"]["integrator"])(self)
        self.compileKernels()

    def step(self, dt):
        self.integrator.step(dt)

    def advance(self, dt):
        """The step is taken: the clock and the count move on."""
        self.nrt += 1
        self.tme += dt
        self.titme = self.tme

    def kernel(self, source, table=None, defines=(), **fixed):
        """One of this case's kernels, from its source: compiled with the
        rest of them, or on its first call if it comes later."""
        bound = BoundKernel(self, source, table, defines, **fixed)
        self.kernels[bound.__name__] = bound
        return bound

    def compileKernels(self):
        """Every kernel made so far that is not yet compiled, at once."""
        pending = [k for k in self.kernels.values() if k.kernel is None]
        self.jit.compile([(k.source, k.defines, k.includes) for k in pending])
        for k in pending:
            k.compile()

    def setConsistify(self):
        """The kernels consistify calls, from what the config picks."""
        eos = self.config["mcPhysics"]["eos"]
        if eos not in ("cpg", "tpg", "realGas"):
            raise pgConfigError("eos", eos)
        self.stateFromPrims = self.kernel(f"thermo/{eos}FromPrims.cpp")
        self.stateFromCons = self.kernel(f"thermo/{eos}FromCons.cpp")

        # Transport properties: the transport and species diffusion choices pick
        # one kernel between them
        if self.config["RHS"]["diffusion"]:
            self.trans = self.kernel(f"transport/{self.mixture.transportKernel}.cpp")
        else:
            self.trans = NullKernel()

        # Switching function between primary and secondary advective fluxes
        #  If we aren't using a secondary flux function, we rely on the
        #  initialization of the switch array "phi" = 0.0 and then
        #  just never change it.
        switch = self.config["RHS"]["switchAdvFlux"]
        if switch is None:
            self.switch = NullKernel()
        else:
            self.switch = self.kernel(f"switches/{switch}.cpp")
            self.phiComm = True

    def setRHS(self):
        """The kernels RHS calls, from what the config picks."""
        rhs = self.config["RHS"]

        self.dQzero = self.kernel("utils/dQzero.cpp")

        # Primary advective fluxes, and how they are applied
        self.primaryAdvFlux = self.kernel(f"advFlux/{rhs['primaryAdvFlux']}.cpp")
        shock = rhs["shockHandling"]
        if shock is None or shock == "artificialDissipation":
            self.applyPrimaryAdvFlux = self.kernel("utils/applyFlux.cpp")
        elif shock == "hybrid":
            self.applyPrimaryAdvFlux = self.kernel(
                "utils/applyHybridFlux.cpp", primary=1.0
            )
        else:
            raise pgConfigError("shockHandling", shock)

        # Secondary advective fluxes
        secondary = rhs["secondaryAdvFlux"]
        if secondary is None:
            self.secondaryAdvFlux = NullKernel()
        else:
            assert (
                shock is not None
            ), "*** You set a secondary flux without a shock handler!"
            self.secondaryAdvFlux = self.kernel(f"advFlux/{secondary}.cpp")
        if shock is None:
            self.applySecondaryAdvFlux = NullKernel()
        elif shock == "artificialDissipation":
            self.applySecondaryAdvFlux = self.kernel("utils/applyFlux.cpp")
        elif shock == "hybrid":
            self.applySecondaryAdvFlux = self.kernel(
                "utils/applyHybridFlux.cpp", primary=0.0
            )

        # spatial derivatives, subgrid mode, diffusive fluxes
        if rhs["diffusion"]:
            self.dqdxyz = self.kernel("utils/dq2FD.cpp")
            sgs = rhs["subgrid"]
            self.sgs = (
                NullKernel() if sgs is None else self.kernel(f"subgrid/{sgs}.cpp")
            )
            self.diffFlux = self.kernel("diffFlux/alphaDampingFlux.cpp")
            self.applyDiffFlux = self.kernel("utils/applyFlux.cpp")
        else:
            self.dqdxyz = NullKernel()
            self.sgs = NullKernel()
            self.diffFlux = NullKernel()
            self.applyDiffFlux = NullKernel()

        sponge = self.config["viscousSponge"]
        if sponge["spongeON"]:
            self.viscousSponge = self.kernel(
                "utils/viscousSponge.cpp",
                origin=sponge["origin"],
                ending=sponge["ending"],
                mult=sponge["multiplier"],
            )
        else:
            self.viscousSponge = NullKernel()

        # Chemical source terms: parked until the mixture package carries reactions
        if self.config["mcPhysics"]["chemistry"]:
            raise pgConfigError("chemistry", True, "Chemistry is not available yet.")
        self.expChem = NullKernel()

    ###########################################################################
    # The boundary faces: what the config makes them, one table, one kernel
    # per hook in the step
    ###########################################################################
    def applyBcValues(self):
        """Make every named face what its config entry says, and give it the
        values that entry sets."""
        bcValues = self.config["bcValues"]

        for blk in self:
            for face in blk.faces:
                if face.bcName is None:
                    continue

                if face.bcName not in bcValues:
                    raise KeyError(
                        f"block {blk.nblki} face {face.nface} carries the name"
                        f" '{face.bcName}', which this config says nothing about."
                        f" It knows {sorted(bcValues)}."
                    )
                entry = bcValues[face.bcName]
                if "bcType" not in entry:
                    raise KeyError(f"bcValues entry '{face.bcName}' names no bcType.")

                face.bcType = entry["bcType"]
                if not getBc(face.bcType).values:
                    continue

                getBc(face.bcType).setValues(face, entry)

    hooks = ("euler", "postEos", "preDqDxyz", "postDqDxyz")

    @property
    def boundaryFaces(self):
        """Every face a boundary condition applies to, in a fixed order."""
        return [face for blk in self for face in blk.faces if getBc(face.bcType).hooks]

    def setBcs(self):
        """A kernel per hook over the faces whose conditions have it: the
        case's conditions are compiled in, and each face's kind at that hook
        picks its own. A hook no condition has is null. Called once the
        faces know their conditions."""
        faces = self.boundaryFaces
        for hook in self.hooks:
            bcTypes = sorted({f.bcType for f in faces if hook in getBc(f.bcType).hooks})
            for face in faces:
                face.kind[hook] = (
                    bcTypes.index(face.bcType) if face.bcType in bcTypes else -1
                )
            if not bcTypes:
                self.bcHooks[hook] = NullKernel()
                continue
            self.bcHooks[hook] = self.kernel(
                f"boundaryConditions/hooks/{hook}.cpp",
                table=lambda hook=hook: self.faceTable(hook),
                defines=("PG_BCS=" + " ".join(f"X({t})" for t in bcTypes),),
                includes=tuple(getBc(t).header(hook) for t in bcTypes),
            )
        self._faceTables = {}
        self.compileKernels()

    def faceTable(self, hook):
        """The records of the faces that have a hook, built once."""
        if hook not in self._faceTables:
            faces = [f for f in self.boundaryFaces if f.kind[hook] >= 0]
            self._faceTables[hook] = Table.ofFaces(faces, hook)
        return self._faceTables[hook]

    def applyBcs(self, hook, faces=None):
        """One hook of the step on every boundary face that has it. The
        euler hook leaves primitives, so the halo's conservatives follow,
        and a condition with more to say once it has a density says it."""
        if not self.bcHooks:
            self.setBcs()
        table = (
            None
            if faces is None
            else Table.ofFaces([f for f in faces if f.kind[hook] >= 0], hook)
        )
        if table is not None and len(table) == 0:
            return
        self.bcHooks[hook](table, tme=self.titme)
        if hook == "euler":
            self.stateFromPrims(
                table=table if table is not None else self.faceTable(hook)
            )
            if not isinstance(self.bcHooks["postEos"], NullKernel):
                self.applyBcs("postEos", faces)
                self.stateFromPrims(
                    table=table if table is not None else self.faceTable("postEos")
                )

    ###########################################################################
    # Making the state consistent, and building its right hand side
    ###########################################################################
    def consistify(self):
        """From the conserved state: halos exchanged, primitives and every
        derived array made consistent, boundary conditions applied."""
        self.communicator.exchange(["Q"])
        self.stateFromCons(nface=-1)
        self._finishConsistify()

    def consistifyFromPrims(self):
        """The same, from the primitive state."""
        self.communicator.exchange(["q"])
        self.stateFromPrims(nface=-1)
        self._finishConsistify()

    def _finishConsistify(self):
        self.applyBcs("euler")
        self.trans(nface=-1)
        self.switch()
        self.viscousSponge()
        if self.phiComm:
            self.communicator.exchange(["phi"])

    def RHS(self):
        """dQ/dt from the current consistent state: every flux difference the
        config asks for, then the chemical source."""
        self.dQzero()

        self.primaryAdvFlux()
        self.applyPrimaryAdvFlux()

        self.secondaryAdvFlux()
        self.applySecondaryAdvFlux()

        if self.config["RHS"]["diffusion"]:
            self.applyBcs("preDqDxyz")
            self.dqdxyz()
            self.communicator.exchange("grads")
            self.applyBcs("postDqDxyz")
            # the subgrid model needs the gradients
            self.sgs()
            self.diffFlux()
            self.applyDiffFlux()

        self.expChem(
            nChemSubSteps=self.config["mcPhysics"]["nChemSubSteps"],
            dt=self.config["timeIntegration"]["dt"],
        )

    ###########################################################################
    # The grid, once every block is on its rank
    ###########################################################################
    def generateHalo(self):
        for blk in self:
            blk.generateHalo()

    def unifyGrid(self):
        self.generateHalo()

        # Lets just be clean and create the edges and corners
        for _ in range(3):
            self.communicator.exchange("nodes")

        for blk in self:
            periodic = [f for f in blk.faces if f.periodicRotation is not None]
            if not periodic:
                continue
            nodes = blk.nodes.get()
            for face in periodic:
                R, t = face.periodicRotation, face.periodicTranslation
                for s0 in face.s0_:
                    # the halo came from the partner, so it lands where the
                    # transform puts it, turned or moved or both
                    p = nodes[s0]
                    nodes[s0] = (p.reshape(-1, 3) @ R.T + t).reshape(p.shape)
            blk.nodes.set(nodes)

    def setBlockCommunication(self):
        for blk in self:
            blk.setBlockCommunication()
        self.communicator = Communicator(self)

    def __repr__(self):
        string = f"  Total blocks: {self.totalBlocks}\n"
        string += f"  Species: {self.thtrdat.speciesNames}\n"
        string += f"  Time Integrator: {self.integrator.integratorName}\n"
        string += f"  Shock Handling: {self.config['RHS']['shockHandling']}\n"
        string += f"  Primary Advective Flux: {self.primaryAdvFlux.__name__}\n"
        string += f"  Switching Function: {self.switch.__name__}\n"
        string += f"  Secondary Advective Flux: {self.secondaryAdvFlux.__name__}\n"
        string += f"  Equation of State: {self.config['mcPhysics']['eos']}\n"
        if self.config["RHS"]["diffusion"]:
            string += f"  Transport Equation: {self.trans.__name__}\n"
        else:
            string += "  Diffusion terms not solved for\n"
        string += f"  Subgrid Model: {self.sgs.__name__}\n"
        if self.config["mcPhysics"]["chemistry"]:
            string += f"  Chemistry mechanism used: {self.expChem.__name__}\n"
            if self.config["mcPhysics"]["nChemSubSteps"] > 1:
                nSub = self.config["mcPhysics"]["nChemSubSteps"]
                string += f"    Number chemical sub steps: {nSub}\n"

        return string
