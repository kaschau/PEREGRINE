#!/usr/bin/env python

from mpi4py import MPI
import kokkos
import peregrinepy as pg
import numpy as np
import matplotlib.pyplot as plt


class state:
    def __init__(self, test):

        self.name = str(test)

        if test == 0:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1
            self.x0 = 0.5

            self.t = 0.2
            self.dt = 1e-4

        if test == 1:
            self.rhoL = 1.0
            self.uL = 0.75
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1
            self.x0 = 0.3

            self.t = 0.2
            self.dt = 1e-4

        if test == 10:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1
            self.x0 = 0.3

            self.t = 0.2
            self.dt = 1e-4

        if test == 2:
            self.rhoL = 1.0
            self.uL = -2.0
            self.pL = 0.4

            self.rhoR = 1.0
            self.uR = 2.0
            self.pR = 0.4
            self.x0 = 0.5

            self.t = 0.15
            self.dt = 1e-4

        if test == 3:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1000.0

            self.rhoR = 1.0
            self.uR = 0.0
            self.pR = 0.01
            self.x0 = 0.5

            self.t = 0.012
            self.dt = 5e-6

        if test == 4:
            self.rhoL = 5.99924
            self.uL = 19.5975
            self.pL = 460.894
            self.x0 = 0.4

            self.rhoR = 5.99242
            self.uR = -6.19633
            self.pR = 46.0950

            self.t = 0.035
            self.dt = 1e-5

        if test == 5:
            self.rhoL = 1.0
            self.uL = -19.5975
            self.pL = 1000.0
            self.x0 = 0.8

            self.rhoR = 1.0
            self.uR = -19.5975
            self.pR = 0.01

            self.t = 0.012
            self.dt = 1e-5

        self.TL = self.pL / (self.rhoL * R)
        self.TR = self.pR / (self.rhoR * R)


results = {
    0: np.array(
        [
            [0.000000, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.020040, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.040080, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.060120, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.080160, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.100200, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.120240, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.140281, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.160321, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.180361, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.200401, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.220441, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.240481, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.260521, 1.000000, 0.000000, 1.000000, 2.500000],
            [0.280561, 0.940866, 0.071685, 0.918203, 2.439782],
            [0.300601, 0.875547, 0.155185, 0.830217, 2.370565],
            [0.320641, 0.813908, 0.238685, 0.749558, 2.302343],
            [0.340681, 0.755790, 0.322186, 0.675712, 2.235118],
            [0.360721, 0.701040, 0.405686, 0.608191, 2.168889],
            [0.380762, 0.649511, 0.489186, 0.546539, 2.103655],
            [0.400802, 0.601057, 0.572687, 0.490322, 2.039418],
            [0.420842, 0.555539, 0.656187, 0.439137, 1.976177],
            [0.440882, 0.512821, 0.739687, 0.392602, 1.913931],
            [0.460922, 0.472773, 0.823188, 0.350359, 1.852682],
            [0.480962, 0.435266, 0.906688, 0.312073, 1.792429],
            [0.501002, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.521042, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.541082, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.561122, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.581162, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.601202, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.621242, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.641283, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.661323, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.681363, 0.426319, 0.927453, 0.303130, 1.777600],
            [0.701403, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.721443, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.741483, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.761523, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.781563, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.801603, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.821643, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.841683, 0.265574, 0.927453, 0.303130, 2.853541],
            [0.861723, 0.125000, 0.000000, 0.100000, 2.000000],
            [0.881764, 0.125000, 0.000000, 0.100000, 2.000000],
            [0.901804, 0.125000, 0.000000, 0.100000, 2.000000],
            [0.921844, 0.125000, 0.000000, 0.100000, 2.000000],
            [0.941884, 0.125000, 0.000000, 0.100000, 2.000000],
            [0.961924, 0.125000, 0.000000, 0.100000, 2.000000],
            [0.981964, 0.125000, 0.000000, 0.100000, 2.000000],
        ]
    ),
    10: np.array(
        [
            [0.005000, 0.999857, 0.000146, 0.999799, 2.499857],
            [0.025000, 0.998750, 0.001461, 0.998252, 2.498752],
            [0.045000, 0.991863, 0.009643, 0.988629, 2.491849],
            [0.065000, 0.961931, 0.045694, 0.947135, 2.461546],
            [0.085000, 0.911061, 0.109067, 0.877796, 2.408719],
            [0.105000, 0.854926, 0.182474, 0.803019, 2.348212],
            [0.125000, 0.800066, 0.258007, 0.731828, 2.286776],
            [0.145000, 0.747208, 0.334808, 0.665059, 2.225147],
            [0.165000, 0.696604, 0.412515, 0.602883, 2.163653],
            [0.185000, 0.648364, 0.490920, 0.545269, 2.102482],
            [0.205000, 0.602540, 0.569835, 0.492114, 2.041832],
            [0.225000, 0.559224, 0.648990, 0.443322, 1.981863],
            [0.245000, 0.518593, 0.727861, 0.398889, 1.922937],
            [0.265000, 0.481096, 0.805058, 0.359152, 1.866321],
            [0.285000, 0.448842, 0.875506, 0.325909, 1.815274],
            [0.305000, 0.428852, 0.921218, 0.305783, 1.782567],
            [0.325000, 0.426689, 0.925936, 0.303770, 1.779809],
            [0.345000, 0.425872, 0.928020, 0.302884, 1.778021],
            [0.365000, 0.425784, 0.928067, 0.302876, 1.778345],
            [0.385000, 0.425367, 0.928088, 0.302856, 1.779969],
            [0.405000, 0.425378, 0.927969, 0.302913, 1.780259],
            [0.425000, 0.425518, 0.927561, 0.303083, 1.780672],
            [0.445000, 0.423273, 0.927849, 0.302960, 1.789386],
            [0.465000, 0.403516, 0.927594, 0.303045, 1.877527],
            [0.485000, 0.327774, 0.927564, 0.303071, 2.311589],
            [0.505000, 0.278382, 0.927568, 0.303063, 2.721642],
            [0.525000, 0.265078, 0.927687, 0.303044, 2.858060],
            [0.545000, 0.264069, 0.927342, 0.303174, 2.870219],
            [0.565000, 0.265038, 0.927482, 0.303172, 2.859710],
            [0.585000, 0.265238, 0.927920, 0.302989, 2.855824],
            [0.605000, 0.265499, 0.926553, 0.303145, 2.854480],
            [0.625000, 0.265417, 0.925613, 0.302920, 2.853255],
            [0.645000, 0.251940, 0.863066, 0.281050, 2.788857],
            [0.665000, 0.126347, 0.011567, 0.101533, 2.008995],
            [0.685000, 0.125002, 0.000000, 0.100002, 2.000010],
            [0.705000, 0.124998, 0.000019, 0.099998, 1.999989],
            [0.725000, 0.125002, -0.000016, 0.100002, 2.000011],
            [0.745000, 0.125000, 0.000020, 0.099999, 1.999997],
            [0.765000, 0.125001, -0.000013, 0.100001, 2.000008],
            [0.785000, 0.124996, 0.000038, 0.099996, 1.999975],
            [0.805000, 0.125006, -0.000039, 0.100006, 2.000037],
            [0.825000, 0.124994, 0.000047, 0.099994, 1.999963],
            [0.845000, 0.125002, -0.000002, 0.100002, 2.000014],
            [0.865000, 0.125001, -0.000012, 0.100001, 2.000009],
            [0.885000, 0.124996, 0.000039, 0.099996, 1.999976],
            [0.905000, 0.125007, -0.000036, 0.100008, 2.000043],
            [0.925000, 0.124995, 0.000028, 0.099995, 1.999969],
            [0.945000, 0.125001, -0.000001, 0.100001, 2.000007],
            [0.965000, 0.125002, -0.000011, 0.100002, 2.000013],
            [0.985000, 0.124998, 0.000019, 0.099998, 1.999988],
            [0.995000, 0.125001, 0.000002, 0.100001, 2.000004],
        ]
    ),
    1: np.array(
        [
            [0.005000, 0.999950, 0.749990, 0.999987, 2.500093],
            [0.025000, 0.999958, 0.749990, 0.999988, 2.500077],
            [0.045000, 0.999965, 0.749992, 0.999989, 2.500061],
            [0.065000, 0.999972, 0.749992, 0.999990, 2.500046],
            [0.085000, 0.999980, 0.749993, 0.999991, 2.500029],
            [0.105000, 0.999988, 0.749995, 0.999991, 2.500007],
            [0.125000, 1.000007, 0.749990, 0.999999, 2.499978],
            [0.145000, 0.999993, 0.750001, 0.999988, 2.499986],
            [0.165000, 0.999876, 0.750136, 0.999828, 2.499882],
            [0.185000, 0.998671, 0.751569, 0.998137, 2.498663],
            [0.205000, 0.988237, 0.763947, 0.983596, 2.488260],
            [0.225000, 0.945762, 0.815494, 0.924980, 2.445064],
            [0.245000, 0.889968, 0.886253, 0.849470, 2.386238],
            [0.265000, 0.833302, 0.961864, 0.774674, 2.324111],
            [0.285000, 0.777646, 1.040153, 0.703225, 2.260749],
            [0.305000, 0.722645, 1.121675, 0.634803, 2.196110],
            [0.325000, 0.668076, 1.208314, 0.568622, 2.127834],
            [0.345000, 0.622062, 1.285825, 0.514489, 2.067676],
            [0.365000, 0.592292, 1.338260, 0.480394, 2.027690],
            [0.385000, 0.581900, 1.357189, 0.468583, 2.013158],
            [0.405000, 0.580165, 1.360195, 0.466726, 2.011179],
            [0.425000, 0.579409, 1.361438, 0.465955, 2.010475],
            [0.445000, 0.579077, 1.362168, 0.465511, 2.009710],
            [0.465000, 0.579199, 1.361915, 0.465667, 2.009960],
            [0.485000, 0.579490, 1.361283, 0.466042, 2.010573],
            [0.505000, 0.579538, 1.360956, 0.466279, 2.011427],
            [0.525000, 0.578447, 1.360870, 0.466335, 2.015465],
            [0.545000, 0.567223, 1.360958, 0.466276, 2.055081],
            [0.565000, 0.492339, 1.360805, 0.466338, 2.367972],
            [0.585000, 0.386363, 1.361308, 0.466091, 3.015889],
            [0.605000, 0.344988, 1.361261, 0.466146, 3.377985],
            [0.625000, 0.337534, 1.360635, 0.466394, 3.454424],
            [0.645000, 0.338213, 1.360521, 0.466377, 3.447362],
            [0.665000, 0.339811, 1.360119, 0.466476, 3.431884],
            [0.685000, 0.339852, 1.360840, 0.466257, 3.429854],
            [0.705000, 0.339687, 1.361045, 0.466114, 3.430463],
            [0.725000, 0.317142, 1.272163, 0.424841, 3.348974],
            [0.745000, 0.127950, 0.026542, 0.103510, 2.022480],
            [0.765000, 0.124998, 0.000004, 0.099998, 1.999987],
            [0.785000, 0.124999, 0.000023, 0.099999, 1.999992],
            [0.805000, 0.125003, -0.000027, 0.100004, 2.000022],
            [0.825000, 0.124995, 0.000054, 0.099994, 1.999964],
            [0.845000, 0.125005, -0.000033, 0.100005, 2.000030],
            [0.865000, 0.124998, 0.000020, 0.099997, 1.999983],
            [0.885000, 0.124998, 0.000026, 0.099998, 1.999988],
            [0.905000, 0.125005, -0.000032, 0.100006, 2.000032],
            [0.925000, 0.124995, 0.000039, 0.099994, 1.999964],
            [0.945000, 0.125005, -0.000028, 0.100006, 2.000032],
            [0.965000, 0.124998, 0.000008, 0.099998, 1.999990],
            [0.985000, 0.124999, 0.000013, 0.099999, 1.999992],
            [0.995000, 0.125002, 0.000001, 0.100002, 2.000011],
        ]
    ),
    2: np.array(
        [
            [0.005000, 1.000005, -1.999987, 0.400006, 1.000009],
            [0.025000, 0.999925, -1.999944, 0.399960, 0.999974],
            [0.045000, 0.998860, -1.999144, 0.399365, 0.999553],
            [0.065000, 0.989331, -1.992008, 0.394053, 0.995756],
            [0.085000, 0.938259, -1.952452, 0.365748, 0.974540],
            [0.105000, 0.874573, -1.899889, 0.330847, 0.945739],
            [0.125000, 0.836552, -1.874149, 0.314713, 0.940507],
            [0.145000, 0.830789, -1.866042, 0.309887, 0.932510],
            [0.165000, 0.834303, -1.863380, 0.307969, 0.922832],
            [0.185000, 0.871710, -1.863425, 0.308292, 0.884160],
            [0.205000, 1.122422, -1.862323, 0.312261, 0.695506],
            [0.225000, 1.463944, -1.859950, 0.315407, 0.538625],
            [0.245000, 1.498810, -1.895114, 0.287515, 0.479572],
            [0.265000, 1.093759, -2.100875, 0.157275, 0.359483],
            [0.285000, 0.746742, -2.273134, 0.092326, 0.309096],
            [0.305000, 0.724390, -2.286426, 0.088300, 0.304740],
            [0.325000, 0.723761, -2.286791, 0.088191, 0.304629],
            [0.345000, 0.723786, -2.286775, 0.088196, 0.304635],
            [0.365000, 0.723754, -2.286798, 0.088189, 0.304625],
            [0.385000, 0.723768, -2.286779, 0.088195, 0.304637],
            [0.405000, 0.723788, -2.286774, 0.088196, 0.304634],
            [0.425000, 0.723760, -2.286796, 0.088190, 0.304623],
            [0.445000, 0.723772, -2.286780, 0.088195, 0.304635],
            [0.465000, 0.723771, -2.286786, 0.088193, 0.304629],
            [0.485000, 0.723767, -2.286789, 0.088192, 0.304628],
            [0.505000, 6.352780, -2.668493, 0.127975, 0.050362],
            [0.525000, -0.006357, -118.614830, 16.289553, -6406.581055],
            [0.545000, -0.074352, 0.518146, -0.040857, 1.373778],
            [0.565000, -1.537567, 2.294805, -0.422861, 0.687549],
            [0.585000, -0.710923, 2.649343, -0.004523, 0.015905],
            [0.605000, -0.720301, 2.649609, -0.004495, 0.015602],
            [0.625000, 0.066998, 0.617548, 0.011935, 0.445345],
            [0.645000, 0.089288, 0.618803, 0.012586, 0.352391],
            [0.665000, 0.120029, 0.679205, 0.016023, 0.333738],
            [0.685000, 0.154391, 0.783042, 0.023166, 0.375117],
            [0.705000, 0.188424, 0.895960, 0.033187, 0.440326],
            [0.725000, 0.223173, 1.006503, 0.045383, 0.508381],
            [0.745000, 0.263135, 1.113436, 0.059905, 0.569148],
            [0.765000, 0.312768, 1.220859, 0.077874, 0.622460],
            [0.785000, 0.373708, 1.330794, 0.100633, 0.673205],
            [0.805000, 0.445339, 1.441421, 0.128967, 0.723979],
            [0.825000, 0.526145, 1.549773, 0.163050, 0.774740],
            [0.845000, 0.614920, 1.654155, 0.202871, 0.824787],
            [0.865000, 0.711475, 1.754608, 0.248700, 0.873889],
            [0.885000, 0.813095, 1.849383, 0.299899, 0.922090],
            [0.905000, 0.903790, 1.925589, 0.347469, 0.961145],
            [0.925000, 0.965360, 1.973838, 0.380809, 0.986183],
            [0.945000, 0.992560, 1.994431, 0.395846, 0.997033],
            [0.965000, 0.999168, 1.999373, 0.399534, 0.999667],
            [0.985000, 0.999952, 1.999970, 0.399974, 0.999982],
            [0.995000, 0.999986, 1.999998, 0.399993, 0.999996],
        ]
    ),
    3: np.array(
        [
            [0.001250, 1.000021, 0.000011, 999.999756, 2499.948242],
            [0.028750, 0.999991, 0.001095, 999.959106, 2499.919434],
            [0.056250, 0.983021, 0.639817, 976.302490, 2482.914551],
            [0.083750, 0.937870, 2.384271, 914.126221, 2436.708252],
            [0.111250, 0.891626, 4.242911, 851.645569, 2387.900146],
            [0.138750, 0.846860, 6.117054, 792.385376, 2339.187988],
            [0.166250, 0.803823, 7.995088, 736.584167, 2290.877441],
            [0.193750, 0.762543, 9.873499, 684.175049, 2243.069824],
            [0.221250, 0.722915, 11.754798, 634.922302, 2195.700928],
            [0.248750, 0.684996, 13.634102, 588.787903, 2148.873535],
            [0.276250, 0.648914, 15.501081, 545.831726, 2102.865967],
            [0.303750, 0.614182, 17.378351, 505.374573, 2057.105225],
            [0.331250, 0.581891, 19.201584, 468.571442, 2013.141846],
            [0.358750, 0.570442, 19.867382, 455.716827, 1997.210205],
            [0.386250, 0.572135, 19.768112, 457.614319, 1999.589600],
            [0.413750, 0.573871, 19.666677, 459.559509, 2002.014526],
            [0.441250, 0.574484, 19.630922, 460.247345, 2002.871826],
            [0.468750, 0.574667, 19.619982, 460.457794, 2003.150879],
            [0.496250, 0.574752, 19.615635, 460.540985, 2003.216064],
            [0.523750, 0.575303, 19.582674, 461.175598, 2004.056763],
            [0.551250, 0.574994, 19.600855, 460.825073, 2003.609375],
            [0.578750, 0.574867, 19.607565, 460.694794, 2003.482666],
            [0.606250, 0.575013, 19.598900, 460.862335, 2003.703735],
            [0.633750, 0.575508, 19.569567, 461.427216, 2004.434326],
            [0.661250, 0.574727, 19.607395, 460.698669, 2003.989868],
            [0.688750, 0.574353, 19.608072, 460.687134, 2005.244263],
            [0.716250, 0.575198, 19.609468, 460.663300, 2002.195679],
            [0.743750, 5.115293, 19.598518, 460.797729, 225.205948],
            [0.771250, 6.006893, 19.594208, 460.965729, 191.848648],
            [0.798750, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.826250, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.853750, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.881250, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.908750, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.936250, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.963750, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.991250, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.996250, 1.000000, 0.000000, 0.010000, 0.025000],
            [0.998750, 1.000000, 0.000000, 0.010000, 0.025000],
        ]
    ),
    4: np.array(
        [
            [0.005000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.035000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.065000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.095000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.125000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.155000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.185000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.215000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.245000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.275000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.305000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.335000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.365000, 5.999240, 19.597500, 460.894012, 192.063522],
            [0.395000, 5.999245, 19.597492, 460.894318, 192.063492],
            [0.425000, 9.998575, 13.086804, 1025.388184, 256.383575],
            [0.455000, 14.127579, 8.792362, 1668.177246, 295.198730],
            [0.485000, 14.319155, 8.677377, 1689.211792, 294.921722],
            [0.515000, 14.256681, 8.676464, 1689.304810, 296.230377],
            [0.545000, 14.255824, 8.693209, 1692.953003, 296.887970],
            [0.575000, 14.322778, 8.732179, 1700.636841, 296.841339],
            [0.605000, 14.185089, 8.704531, 1693.077881, 298.390442],
            [0.635000, 14.138174, 8.687351, 1689.331299, 298.718109],
            [0.665000, 14.813260, 8.671492, 1691.249878, 285.428375],
            [0.695000, 21.056679, 8.681622, 1688.045044, 200.416824],
            [0.725000, 27.033966, 8.678861, 1688.877441, 156.181076],
            [0.755000, 29.811007, 8.673041, 1687.724487, 141.535355],
            [0.785000, 30.895372, 8.704525, 1700.081177, 137.567627],
            [0.815000, 30.916443, 8.622582, 1674.494263, 135.404831],
            [0.845000, 5.992420, -6.196331, 46.095009, 19.230549],
            [0.875000, 5.992420, -6.196330, 46.095005, 19.230547],
            [0.905000, 5.992420, -6.196330, 46.095005, 19.230547],
            [0.935000, 5.992420, -6.196330, 46.095005, 19.230547],
            [0.965000, 5.992420, -6.196330, 46.095005, 19.230547],
            [0.975000, 5.992420, -6.196330, 46.095005, 19.230547],
            [0.995000, 5.992420, -6.196330, 46.095005, 19.230547],
        ]
    ),
    5: np.array(
        [
            [0.001000, 1.000049, -19.596025, 1000.052307, 2500.008057],
            [0.037000, 1.000007, -19.597523, 999.996033, 2499.972900],
            [0.073000, 1.000006, -19.597351, 1000.002686, 2499.992920],
            [0.109000, 0.998421, -19.538214, 997.790344, 2498.420166],
            [0.145000, 0.944755, -17.482954, 923.515930, 2443.797119],
            [0.181000, 0.883854, -15.034174, 841.256104, 2379.512207],
            [0.217000, 0.825703, -12.566699, 764.804871, 2315.616943],
            [0.253000, 0.770594, -10.096546, 694.310303, 2252.515137],
            [0.289000, 0.718513, -7.628442, 629.514282, 2190.338379],
            [0.325000, 0.669352, -5.163714, 570.050232, 2129.113525],
            [0.361000, 0.622936, -2.700000, 515.488220, 2068.783447],
            [0.397000, 0.580862, -0.336256, 467.411438, 2011.715698],
            [0.433000, 0.573999, 0.062075, 459.699005, 2002.177246],
            [0.469000, 0.574569, 0.028728, 460.339569, 2002.978882],
            [0.505000, 0.574908, 0.008661, 460.726624, 2003.479004],
            [0.541000, 0.575147, -0.004446, 460.978668, 2003.742676],
            [0.577000, 0.574993, 0.004282, 460.810089, 2003.547607],
            [0.613000, 0.575045, 0.001015, 460.874176, 2003.643799],
            [0.649000, 0.575020, 0.001832, 460.859650, 2003.667603],
            [0.685000, 0.575028, 0.001562, 460.863403, 2003.656738],
            [0.721000, 0.575153, -0.006795, 461.023895, 2003.920044],
            [0.757000, 0.574934, 0.001333, 460.867737, 2004.002441],
            [0.793000, 1.088747, 0.001866, 460.854218, 1058.221924],
            [0.829000, 5.998665, -0.000554, 460.849335, 192.063293],
            [0.865000, 1.000001, -19.597448, 0.009967, 0.024918],
            [0.901000, 0.999999, -19.597433, 0.010052, 0.025131],
            [0.937000, 1.000000, -19.597437, 0.010034, 0.025085],
            [0.973000, 1.000001, -19.597427, 0.010046, 0.025116],
            [0.999000, 1.000000, -19.597450, 0.009998, 0.024994],
        ]
    ),
}

R = 281.4583333333333
gamma = 1.4


def simulate(testnum, index="i"):

    test = state(testnum)
    print("State {}".format(testnum))
    print("--------------------------")
    print("Left State")
    print("PL = {}".format(test.pL))
    print("TL = {}".format(test.TL))
    print("rhoL = {}".format(test.rhoL))
    print("uL = {}".format(test.uL))
    print("--------------------------")
    print("Right State")
    print("PR = {}".format(test.pR))
    print("TR = {}".format(test.TR))
    print("rhoR = {}".format(test.rhoR))
    print("uR = {}".format(test.uR))
    print("--------------------------")

    nx = 201
    config = pg.files.configFile()
    config["thermochem"]["spdata"] = ["DB"]
    config["RHS"]["shockHandling"] = "hybrid"
    config["RHS"]["primaryAdvFlux"] = "secondOrderKEEP"
    config["RHS"]["secondaryAdvFlux"] = "muscl2hllc"
    config["RHS"]["switchAdvFlux"] = "vanLeer"
    # config["RHS"]["primaryAdvFlux"] = "muscl2hllc"
    config["solver"]["timeIntegration"] = "rk4"
    mb = pg.multiBlock.generateMultiBlockSolver(1, config)
    print(mb)

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    dimsPerBlock = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)

    pg.grid.create.multiBlockCube(
        mb,
        mbDims=[1, 1, 1],
        dimsPerBlock=dimsPerBlock,
        lengths=lengths,
    )

    mb.initSolverArrays(config)

    blk = mb[0]
    ng = blk.ng

    for face in blk.faces:
        face.bcType = "adiabaticSlipWall"

    mb.setBlockCommunication()
    mb.unifyGrid()
    mb.computeMetrics(fdOrder=2)

    ccArray = {"i": "xc", "j": "yc", "k": "zc"}
    uIndex = {"i": 1, "j": 2, "k": 3}
    xc = blk.array[ccArray[index]]
    # Initialize Left/Right properties
    blk.array["q"][:, :, :, 0] = np.where(xc <= test.x0, test.pL, test.pR)
    blk.array["q"][:, :, :, uIndex[index]] = np.where(xc <= test.x0, test.uL, test.uR)
    blk.array["q"][:, :, :, 4] = np.where(xc <= test.x0, test.TL, test.TR)

    # Update boundary conditions
    if index == "i":
        lowFace = 1
        highFace = 2
    elif index == "j":
        lowFace = 3
        highFace = 4
    elif index == "k":
        lowFace = 5
        highFace = 6

    face = blk.getFace(lowFace)
    if test.uL == 0.0:
        pass
    else:
        face.array["qBcVals"] = np.zeros((blk.array["q"][face.s1_].shape))
        inputBcValues = {}
        if test.uL > 0:
            face.bcType = "constantVelocitySubsonicInlet"
            bcVelo = rotate([test.uL, 0.0, 0.0], index)
            inputBcValues["u"] = bcVelo[0]
            inputBcValues["v"] = bcVelo[1]
            inputBcValues["w"] = bcVelo[2]
            inputBcValues["T"] = test.TL
            pg.bcs.prepInlets.prep_constantVelocitySubsonicInlet(
                blk, face, inputBcValues
            )
        elif test.uL < 0:
            face.bcType = "constantPressureSubsonicExit"
            inputBcValues["p"] = test.pL
            pg.bcs.prepExits.prep_constantPressureSubsonicExit(blk, face, inputBcValues)
        shape = blk.array["q"][face.s1_].shape
        pg.misc.createViewMirrorArray(face, "qBcVals", shape)

    face = blk.getFace(highFace)
    if test.uR == 0.0:
        pass
    else:
        face.array["qBcVals"] = np.zeros((blk.array["q"][face.s1_].shape))
        inputBcValues = {}
        if test.uR < 0:
            face.bcType = "constantVelocitySubsonicInlet"
            bcVelo = rotate([test.uR, 0.0, 0.0], index)
            inputBcValues["u"] = bcVelo[0]
            inputBcValues["v"] = bcVelo[1]
            inputBcValues["w"] = bcVelo[2]
            inputBcValues["T"] = test.TR
            pg.bcs.prepInlets.prep_constantVelocitySubsonicInlet(
                blk, face, inputBcValues
            )
        elif test.uR > 0:
            face.bcType = "constantPressureSubsonicExit"
            inputBcValues["p"] = test.pR
        pg.bcs.prepExits.prep_constantPressureSubsonicExit(blk, face, inputBcValues)
        shape = blk.array["q"][face.s1_].shape
        pg.misc.createViewMirrorArray(face, "qBcVals", shape)

    # Update cons
    mb.eos(blk, mb.thtrdat, 0, "prims")
    pg.consistify(mb)

    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    x = blk.array[ccArray[index]][s_]
    rho = blk.array["Q"][s_][:, 0]
    p = blk.array["q"][s_][:, 0]
    phi = blk.array["phi"][s_][:, uIndex[index] - 1]
    u = blk.array["q"][s_][:, uIndex[index]]

    while mb.tme < test.t:
        pg.misc.progressBar(mb.tme, test.t)
        mb.step(test.dt)

    fig, ax1 = plt.subplots()
    ax1.set_title(f"{mb.tme:.2f}")
    ax1.set_xlabel(r"x")
    ax1.plot(x, phi, "--", color="gold", label="phi", linewidth=0.25)
    ax1.plot(x, rho, color="g", label="rho", linewidth=0.5)
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax1.plot(x, u, color="k", label="u", linewidth=0.5)

    res = results[testnum]
    rx = res[:, 0]
    rrho = res[:, 1]
    ru = res[:, 2]
    rp = res[:, 3]
    ax1.scatter(rx, rrho, color="g", label="Analyticsl", marker="o", s=0.2)
    ax1.scatter(rx, rp, color="r", marker="o", s=0.2)
    ax1.scatter(rx, ru, color="k", marker="o", s=0.2)

    ax1.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        kokkos.initialize()
        testnum = 0
        index = "i"
        simulate(testnum, index)
        kokkos.finalize()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
