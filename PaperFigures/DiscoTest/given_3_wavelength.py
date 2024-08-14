import lightweaver as lw
from matplotlib import pyplot as plt
import numpy as np
from lightweaver.atomic_model import AtomicModel, AtomicLevel, VoigtLine, LineType
from lightweaver.LwCompiled import BackgroundProvider

SIZE = 1024
FLATLAND = False

class MyLine(VoigtLine):
    @property
    def Aji(self) -> float:
        return 4.0 * np.pi * self.lambda0_m / lw.HC

    @property
    def Bji(self) -> float:
        # NOTE(cmo): No stimulated emission.
        return 0.0

    @property
    def Bij(self) -> float:
        return 4.0 * np.pi * self.lambda0_m / lw.HC

MyAtom = lambda: \
AtomicModel(element=lw.Element(Z=1),
    levels=[
        AtomicLevel(E=0.00, g=1, label="Ground", stage=0),
        AtomicLevel(E=0.01, g=1, label="L1", stage=0),
        AtomicLevel(E=0.02, g=1, label="L2", stage=0),
        AtomicLevel(E=0.03, g=1, label="L3", stage=0),
    ],
    lines=[
        # MyLine(j=1, i=0, f=1.0, type=LineType.CRD, quadrature=TabulatedQuadrature([0.0]), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.7e+08)], elastic=[])),
        # MyLine(j=2, i=0, f=0.5**2, type=LineType.CRD, quadrature=TabulatedQuadrature([0.0]), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=9.98e+07)], elastic=[])),
        # MyLine(j=3, i=0, f=1.0/3.0**2, type=LineType.CRD, quadrature=TabulatedQuadrature([0.0]), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=9.98e+07)], elastic=[])),
    ],
    continua=[
    ],
    collisions=[
])

class FixedXBc(lw.BoundaryCondition):
    def __init__(self, mode):
        modes = ['lower', 'upper']
        if not any(mode == m for m in modes):
            raise ValueError('Invalid mode')

        self.mode = mode
        # NOTE(cmo): This data needs to be in (mu, toObs) order, i.e. mu[0]
        # down, mu[0] up, mu[1] down...
        # self.I = I1d.reshape(I1d.shape[0], -1, I1d.shape[-1])
        self.I = None

    def set_bc(self, I1d):
        # Shape [Nwave, Nmu, Nspace]
        self.I = I1d

    def compute_bc(self, atmos, spect):
        # if spect.wavelength.shape[0] != self.I.shape[0]:
        #     result = np.ones((spect.wavelength.shape[0], spect.I.shape[1], atmos.Nz))
        # else:
        if self.I is None:
            raise ValueError('I has not been set (x%sBc)' % self.mode)
        result = np.copy(self.I)
        return result

def draw_disk(arr, centre, radius, color):
    X, Y = np.mgrid[:SIZE, :SIZE]
    dist_x = X - centre[0]
    dist_y = Y - centre[1]

    mask = ((dist_x**2 + dist_y**2) <= radius * radius).flatten()
    arr[:, mask] = color[:, None]

class MyBackground(BackgroundProvider):
    def __init__(self, eq_pops, rad_set, wavelength):
        if wavelength.shape[0] != 3:
            raise ValueError("Currently only expecting 3 wavelengths")

    def compute_background(self, atmos, chi, eta, sca):
        sca[...] = 0.0

        centre = int(SIZE // 2)
        c = np.array([centre+400, centre])
        color = np.array([0.0, 0.0, 2.0])
        draw_disk(eta, c, 40, color)

        c = np.array([centre-400, centre])
        color = np.array([2.0, 0.0, 0.0])
        draw_disk(eta, c, 40, color)

        c = np.array([centre, centre-400])
        color = np.array([0.0, 0.0, 0.5])
        draw_disk(eta, c, 40, color)

        c = np.array([centre, centre+400])
        color = np.array([0.5, 0.0, 0.0])
        draw_disk(eta, c, 40, color)

        c = np.array([200, 200])
        color = np.array([0.0, 3.0, 0.0])
        draw_disk(eta, c, 40, color)

        c = np.array([SIZE-200, 200])
        color = np.array([0.0, 0.0, 3.0])
        draw_disk(eta, c, 40, color)

        c = np.array([400, 700])
        color = np.array([3.0, 0.0, 3.0])
        draw_disk(eta, c, 40, color)

        chi[...] = 1e-10
        bg = 1e-10
        c = np.array([centre+400, centre])
        color = np.array([bg, bg, 0.5])
        draw_disk(chi, c, 40, color)

        c = np.array([centre-400, centre])
        color = np.array([0.5, bg, bg])
        draw_disk(chi, c, 40, color)

        c = np.array([centre, centre-400])
        color = np.array([bg, bg, 0.5])
        draw_disk(chi, c, 40, color)

        c = np.array([centre, centre+400])
        color = np.array([0.5, bg, bg])
        draw_disk(chi, c, 40, color)

        c = np.array([centre+340, centre])
        color = np.array([0.2, 0.2, 0.2])
        draw_disk(chi, c, 6, color)

        c = np.array([centre-340, centre])
        color = np.array([0.2, 0.2, 0.2])
        draw_disk(chi, c, 6, color)

        box_size = 250
        cc = chi.reshape(-1, SIZE, SIZE)
        chi_r = 1e-4
        cc[0, centre-box_size:centre+box_size, centre-box_size:centre+box_size] = chi_r
        cc[1:, centre-box_size:centre+box_size, centre-box_size:centre+box_size] = 1e2 * chi_r

        c = np.array([200, 200])
        color = np.array([bg, 1.0, bg])
        draw_disk(chi, c, 40, color)

        c = np.array([SIZE-200, 200])
        color = np.array([bg, bg, 1.0])
        draw_disk(chi, c, 40, color)

        c = np.array([400, 700])
        color = np.array([1.0, bg, 1.0])
        draw_disk(chi, c, 40, color)



atom = MyAtom()
atmos = lw.Atmosphere.make_2d(
    # height=np.arange(0, SIZE, dtype=np.float64),
    height=np.arange(SIZE-1, -1, -1, dtype=np.float64),
    x=np.arange(0, SIZE, dtype=np.float64),
    temperature=np.ones((SIZE, SIZE)) * 6000,
    vx=np.zeros((SIZE, SIZE)),
    vz=np.zeros((SIZE, SIZE)),
    vturb=np.zeros((SIZE, SIZE)),
    ne=np.ones((SIZE, SIZE)),
    nHTot=np.ones((SIZE, SIZE))*100,
    xLowerBc=FixedXBc('lower'),
    xUpperBc=FixedXBc('upper'),
    zLowerBc=lw.ZeroRadiation(),
    zUpperBc=lw.ZeroRadiation(),
)
atmos.quadrature(10)
if FLATLAND:
    sin_phi = np.sqrt(1.0 - atmos.muy**2)
    atmos.muz[:] = atmos.muz[:] / sin_phi
    atmos.mux[:] = atmos.mux[:] / sin_phi
    atmos.muy[:] = 0.0
    print(atmos.mux**2 + atmos.muy**2 + atmos.muz**2)

a_set = lw.RadiativeSet([MyAtom()])
a_set.set_active("H")
spect = a_set.compute_wavelength_grid(extraWavelengths=[500.0, 600.0, 700.0])
eq_pops = a_set.compute_eq_pops(atmos)

ctx = lw.Context(atmos, spect, eq_pops, backgroundProvider=MyBackground, formalSolver='piecewise_besser_2d', interpFn='interp_besser_2d')
# ctx = lw.Context(atmos, spect, eq_pops, backgroundProvider=MyBackground)
atmos.xLowerBc.set_bc(np.zeros((3, atmos.muz.shape[0], SIZE)))
atmos.xUpperBc.set_bc(np.zeros((3, atmos.muz.shape[0], SIZE)))

plt.ion()
# eta = np.moveaxis(np.copy(ctx.background.eta).reshape(3, SIZE, SIZE), 0, -1)
# plt.imshow(eta, origin="lower", interpolation="nearest")

ctx.formal_sol_gamma_matrices()

J = np.moveaxis(np.copy(ctx.spect.J).reshape(3, SIZE, SIZE), 0, -1)
np.save("J_10th.npy", J)