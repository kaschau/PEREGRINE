import numpy as np

from .restart import restart
from .solverBlock import solverBlock
from ..mpiComm import Communicator


class solver(restart):
    """A list of peregrinepy.multiBlock.solver.
    Inherits from peregrinepy.multiBlock.restart"""

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
        self.titme = 0.0
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

        # Halo exchange, once the blocks know their neighbors
        self.communicator = None

    def generateHalo(self):
        for blk in self:
            blk.generateHalo()

    def unifyGrid(self):
        self.generateHalo()

        # Lets just be clean and create the edges and corners
        for _ in range(3):
            self.communicator.exchange(["x", "y", "z"])

        # Device is up to date after communicate, so pull back down
        for blk in self:
            blk.updateHostView(["x", "y", "z"])

        for blk in self:
            for face in blk.faces:
                if face.periodicRotation is None:
                    continue
                R, t = face.periodicRotation, face.periodicTranslation
                for s0 in face.s0_:
                    # the halo came from the partner, so it lands where the
                    # transform puts it, turned or moved or both
                    p = np.column_stack(
                        [blk.array[v][s0].ravel() for v in ("x", "y", "z")]
                    )
                    p = p @ R.T + t
                    for n, v in enumerate(("x", "y", "z")):
                        blk.array[v][s0] = p[:, n].reshape(blk.array[v][s0].shape)

        # Push back up the device
        for blk in self:
            blk.updateDeviceView(["x", "y", "z"])

    def setBlockCommunication(self):
        for blk in self:
            blk.setBlockCommunication()
        self.communicator = Communicator(self)

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
