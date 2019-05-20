
import numpy as np
from unittest import TestCase
from lasso.dyna.D3plot import D3plot


class D3plotTest(TestCase):

    def test_init(self):
        d3plot = D3plot()
        d3plot = D3plot("test/simple_d3plot/d3plot")

    def test_header(self):

        test_header_data = {
            'title': '                                        ',
            'runtime': 1472027823,
            'filetype': 1,
            'source_version': -971095103,
            'release_version': 'R712',
            'version': 960.0,
            'ndim': 3,
            'numnp': 4915,
            'it': 0,
            'iu': 1,
            'iv': 1,
            'ia': 1,
            'nel2': 0,
            'nel4': 4696,
            'nel8': 0,
            'nelth': 0,
            'nel20': 0,
            'nel27': 0,
            'nel48': 0,
            'nv1d': 6,
            'nv2d': 102,
            'nv3d': 32,
            'nv3dt': 90,
            'nummat4': 1,
            'nummat8': 0,
            'nummat2': 0,
            'nummatt': 0,
            'icode': 6,
            'nglbv': 13,
            'numst': 0,
            'numds': 0,
            'neiph': 25,
            'neips': 19,
            'maxint': 3,
            'nmsph': 0,
            'ngpsph': 0,
            'narbs': 9624,
            'ioshl1': 1,
            'ioshl2': 1,
            'ioshl3': 1,
            'ioshl4': 1,
            'ialemat': 0,
            'ncfdv1': 0,
            'ncfdv2': 0,
            'nadapt': 0,
            'nmmat': 1,
            'numfluid': 0,
            'inn': 1,
            'npefg': 0,
            'idtdt': 0,
            'extra': 0,
            'nt3d': 0,
            'neipb': 0,
            'external_numbers_dtype': np.int32,
            'mattyp': 0,
            'has_rigid_road_surface': False,
            'has_rigid_body_data': False,
            'has_temperatures': False,
            'has_mass_scaling': False,
            'has_nel10': False,
            'mdlopt': 2,
            'istrn': 1,
            'has_internal_energy': True,
            'numrbe': 0,
            'num_sph_vars': 0,
            'numbering_header': {
                'nsort': 25894,
                'nsrh': 30809,
                'nsrb': 30809,
                'nsrs': 30809,
                'nsrt': 35505,
                'nsortd': 4915,
                'nsrhd': 0,
                'nsrbd': 0,
                'nsrsd': 4696,
                'nsrtd': 0
            },
            'ntype': [90001, 90000],
            'title2': '                                                                        ',
            'n_timesteps': 1
        }

        d3plot = D3plot("test/simple_d3plot/d3plot")
        header = d3plot.header
        self.assertEqual(header["title"], " "*40)

        for name, value in test_header_data.items():
            self.assertEqual(header[name], value, "Invalid var %s" % name)
