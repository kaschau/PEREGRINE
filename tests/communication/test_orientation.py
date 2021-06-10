import unittest
import peregrinepy as pg
import numpy as np

class TestOrientation(unittest.TestCase):

    def test_123(self):
        config = pg.files.config_file()
        config['RunTime']['ngls'] = 2
        mb = pg.multiblock(2,config)

        pg.grid.create.multiblock_cube(mb, mb_dimensions=[2,1,1])
        mb[0].connectivity['2']['comm_rank'] = 0
        mb[1].connectivity['1']['comm_rank'] = 0
        pg.mpicomm.blockcomm.set_block_communication(mb,config)

        shape = mb[0].array['x'].shape

        for blk in mb:
            blk.array['x'] = np.random.random((shape))
            blk.array['y'] = np.random.random((shape))
            blk.array['z'] = np.random.random((shape))
        pg.mpicomm.blockcomm.communicate(mb,['x','y','z'])

        passfail = []
        for var in ['x','y','z']:
            comare = np.equal(mb[0].array[var][mb[0].slice_s3['2']], mb[1].array[var][mb[1].slice_r3['1']])
            passfail.append(np.all(comare))
            compare = np.equal(mb[1].array[var][mb[1].slice_s3['1']], mb[0].array[var][mb[0].slice_r3['2']])
            passfail.append(np.all(comare))

        self.assertTrue(False not in passfail)


if __name__ == '__main__':
    unittest.main()
