from .restart import restart
from .solverBlock import solverBlock
from .. import mpiComm


class solver(restart):
    """A list of peregrinepy.multiBlock.solver.
    Inherits from peregrinepy.multiBlock.restart"""

    __slots__ = (
        "config",
        "kokkosSpace" "thtrdat",
        "eos",
        "trans",
        "dqdxyz",
        "primaryAdvFlux",
        "applyPrimaryAdvFlux",
        "switch",
        "secondaryAdvFlux",
        "applySecondaryAdvFlux" "diffFlux",
        "applyDiffFlux",
        "expChem",
        "impChem",
        "resultsWriter",
    )

    mbType = "solver"
    hasConservatives = True

    def _newBlock(self, nblki):
        return solverBlock(nblki, self.speciesNames, self.ng, self.config)

    def progress(self, n, message):
        """A running case reports through its own machinery, not a bar."""

    def __init__(self, nblks, spNames, ng, config):
        assert isinstance(spNames, list), f"spNames must me a list not {type(spNames)}"

        self.ng = ng

        self.config = config
        temp = [solverBlock(i, spNames, ng, config) for i in range(nblks)]
        super().__init__(nblks, spNames, temp)

        # time integrator time
        self._titme = 0.0
        # Save the species data
        self.thtrdat = None

        #########################################
        # Consistify
        #########################################
        # We need the following in order to use
        # consisify method
        self.phiComm = False
        self.eos = None
        self.trans = None

        #########################################
        # RHS
        #########################################
        # We need the following in order to use
        # RHS method
        self.primaryAdvFlux = None
        self.applyPrimaryAdvFlux = None
        self.secondaryAdvFlux = None
        self.applySecondaryAdvFlux = None
        self.switch = None

        self.dqdxyz = None
        self.diffFlux = None
        self.applyDiffFlux = None
        self.viscousSponge = None

        # Explicit chemistry is solved for in RHS,
        #  so we want to keep implicit chemistry
        #  separate
        self.expChem = None
        self.impChem = None

        # Result output
        self.resultsWriter = None

    @property
    def titme(self):
        return self._titme

    @titme.setter
    def titme(self, val):
        self._titme = val
        for blk in self:
            blk.titme = val

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()

    def unifyGrid(self):
        self.generateHalo()

        # Lets just be clean and create the edges and corners
        for _ in range(3):
            mpiComm.communicate(self, ["x", "y", "z"])

        # Device is up to date after communicate, so pull back down
        for blk in self:
            blk.updateHostView(["x", "y", "z"])

        for blk in self:
            for face in blk.faces:
                bc = face.bcType
                if not bc.startswith("periodic"):
                    continue
                for i, s0 in enumerate(face.s0_):
                    x = blk.array["x"][s0]
                    y = blk.array["y"][s0]
                    z = blk.array["z"][s0]

                    # Translate periodics
                    if face.bcType == "periodicTransLow":
                        x[:] -= face.periodicAxis[0] * face.periodicSpan
                        y[:] -= face.periodicAxis[1] * face.periodicSpan
                        z[:] -= face.periodicAxis[2] * face.periodicSpan
                    elif face.bcType == "periodicTransHigh":
                        x[:] += face.periodicAxis[0] * face.periodicSpan
                        y[:] += face.periodicAxis[1] * face.periodicSpan
                        z[:] += face.periodicAxis[2] * face.periodicSpan
                    elif face.bcType.startswith("periodicRot"):
                        if face.bcType == "periodicRotLow":
                            rotM = face.array["periodicRotMatrixDown"]
                        elif face.bcType == "periodicRotHigh":
                            rotM = face.array["periodicRotMatrixUp"]
                        tempx = (
                            rotM[0, 0] * x[:] + rotM[0, 1] * y[:] + rotM[0, 2] * z[:]
                        )
                        tempy = (
                            rotM[1, 0] * x[:] + rotM[1, 1] * y[:] + rotM[1, 2] * z[:]
                        )
                        tempz = (
                            rotM[2, 0] * x[:] + rotM[2, 1] * y[:] + rotM[2, 2] * z[:]
                        )
                        x[:] = tempx[:]
                        y[:] = tempy[:]
                        z[:] = tempz[:]

        # Push back up the device
        for blk in self:
            blk.updateDeviceView(["x", "y", "z"])

    def setBlockCommunication(self):
        for blk in self:
            blk.setBlockCommunication()

    def __repr__(self):
        string = f"  Total blocks: {self.totalBlocks}\n"
        string += f"  Species: {self.thtrdat.speciesNames}\n"
        string += f"  Time Integrator: {self.integratorName}\n"
        string += f"  Shock Handling: {self.config['RHS']['shockHandling']}\n"
        string += f"  Primary Advective Flux: {self.primaryAdvFlux.__name__}\n"
        string += f"  Switching Function: {self.switch.__name__}\n"
        string += f"  Secondary Advective Flux: {self.secondaryAdvFlux.__name__}\n"
        string += f"  Equation of State: {self.eos.__name__}\n"
        if self.config["RHS"]["diffusion"]:
            string += f"  Transport Equation: {self.trans.__name__}\n"
        else:
            string += "  Diffusion terms not solved for\n"
        string += f"  Subgrid Model: {self.sgs.__name__}\n"
        if self.config["thermochem"]["chemistry"]:
            string += f"  Explicit chemistry mechanism used: {self.expChem.__name__}\n"
            if self.config["thermochem"]["nChemSubSteps"] > 1:
                nSub = self.config["thermochem"]["nChemSubSteps"]
                string += f"    Number chemical sub steps: {nSub}\n"
            string += f"  Implicit chemistry mechanism used: {self.impChem.__name__}\n"

        return string
