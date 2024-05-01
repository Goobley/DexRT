from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import promweaver as pw
import lightweaver as lw
import astropy.constants as const
import astropy.units as u
from numpy.polynomial.legendre import leggauss

def compute_intersection_t(p0, d):
    # NOTE(cmo): Units of Rs
    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - 1.0
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    return (-b - np.sqrt(delta)) / (2.0 * a)

def compute_intersection_length(p0, d):
    # NOTE(cmo): Units of Rs
    a = 1.0
    b = 2.0 * (p0 @ d)
    c = p0 @ p0 - 1.0
    delta = b*b - 4.0 * a * c

    if delta < 0:
        return None
    t1 = (-b - np.sqrt(delta)) / (2.0 * a)
    t2 = (-b + np.sqrt(delta)) / (2.0 * a)
    return t2 - t1

Rs = np.float64(const.R_sun.value)
def compute_intersection_angle(p0, d):
    # NOTE(cmo): Units of Rs
    p0 = p0.copy()
    p0 /= Rs
    p0[2] += 1.0
    t_min = compute_intersection_t(p0, d)
    if t_min is None:
        return None
    int_loc = p0 + t_min * d
    # cosphi = -(d @ int_loc) / Rs
    # NOTE(cmo): Rs is 1.0
    costheta = -(d @ int_loc)
    return costheta

def classical_dilution(h):
    return 0.5 * (1.0 - np.sqrt(1.0 - Rs**2 / (Rs + h)**2))

def format_fp_array_cpp(arr):
    res = "{"
    for i, x in enumerate(arr):
        res += f"FP({x})"
        if i != arr.shape[0]-1:
            res += ", "
    res += "}"
    return res


if __name__ == "__main__":
    bc_ctx = pw.compute_falc_bc_ctx(["Ca"])
    orig_bc_table = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    I_with_zero = np.zeros((orig_bc_table["I"].shape[0], orig_bc_table["I"].shape[1] + 1))
    I_with_zero[:, 1:] = orig_bc_table["I"][...]
    orig_bc_table["I"] = I_with_zero
    orig_bc_table["mu_grid"] = np.concatenate([[0], orig_bc_table["mu_grid"]])
    bc_table = orig_bc_table

    bc_provider = pw.TabulatedPromBcProvider(**bc_table)

    Altitude = 30e6

    uni_J = pw.UniformJPromBc("filament", bc_provider=bc_provider, altitude_m=Altitude, Nrays=400)

    dummy_atmos = lw.Atmosphere.make_1d(
        lw.ScaleType.Geometric,
        np.linspace(0, 1.0, 2)[::-1],
        np.ones(2) * 3000,
        vlos=np.zeros(2),
        vturb=np.zeros(2),
        ne=np.ones(2) * 1e10,
        nHTot=np.ones(2) * 1e12,
        lowerBc=uni_J
    )
    dummy_atmos.quadrature(5)

    @dataclass
    class DummySpect:
        wavelength: np.ndarray

    def gauss_radau(n):
        # https://mathworld.wolfram.com/RadauQuadrature.html

        root_arg = [0.0] * (n + 1)
        root_arg[-1] = 1.0
        root_arg[-2] = 1.0
        x_i = np.polynomial.legendre.legroots(root_arg)[1:]

        def Pnm1(x_i):
            poly_arg = [0.0] * n
            poly_arg[-1] = 1.0
            return np.polynomial.legendre.legval(x_i, poly_arg)

        w_i = (1.0 - x_i) / (n**2 * Pnm1(x_i)**2)
        x_i = np.concatenate(([-1], x_i))
        w_i = np.concatenate(([2.0 /  n**2], w_i))
        return x_i, w_i

    USE_GAUSS_RADAU = True

    spect = DummySpect(np.array([100.0, 394.0]))
    Jbc = uni_J.compute_bc(dummy_atmos, spect)

    Nrays = 1024
    Nincl = 8
    if not USE_GAUSS_RADAU:
        incl_muy, incl_wmu = leggauss(Nincl)
    else:
        incl_muy, incl_wmu = gauss_radau(Nincl)
    incl_muy = 0.5 * incl_muy + 0.5
    incl_wmu *= 0.5



    Js = []
    Juniform = []
    alts = np.linspace(10e6, 20e6, 16)
    x_size = 2e6
    grid = np.zeros((alts.shape[0], 16))
    for alt_idx, alt in enumerate(alts):
        for x_idx in range(16):
            x_pos = ((x_idx + 0.5) - 8) * (10e6 / 16)
            J = np.zeros_like(spect.wavelength)
            J_uni = 0.0
            for phi_idx in range(Nrays):
                phi = (phi_idx + 0.5) * 2.0 * np.pi / Nrays
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                start_pos = np.array([x_pos, 0.0, alt])
                wphi = 1.0 / Nrays
                if (sin_phi < 0.0):
                    for theta_idx in range(Nincl):
                        cos_theta = incl_muy[theta_idx]
                        sin_theta = np.sqrt(1.0 - cos_theta**2)
                        mu = np.array([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta])
                        # mu = np.array([cos_phi, 0.0, sin_phi])
                        mu_atmos = compute_intersection_angle(start_pos, mu)
                        if mu_atmos is not None:
                            J_uni += 1.0 * wphi * incl_wmu[theta_idx]
                            J += bc_provider.compute_I(spect.wavelength, mu_atmos) * wphi * incl_wmu[theta_idx]
            if x_idx == 0:
                Js.append(J)
                Juniform.append(J_uni)
            grid[alt_idx, x_idx] = J[0]

    Js = np.array(Js)
    Juniform = np.array(Juniform)

    Jbcs = []
    for alt in alts:
        uni_J.altitude = alt
        uni_J.update_bc(dummy_atmos, spect)
        Jbcs.append(uni_J.compute_bc(dummy_atmos, spect)[:, 0, 0])
    Jbcs = np.array(Jbcs)
    # NOTE(cmo): Because Jbc is only the rays going into the bottom of the filament, and not actually J, it's too large by a factor of 2 -- need to include downward rays -- assumed to be 0.
    Jbcs *= 0.5

    plt.ion()
    # plt.plot(alts, classical_dilution(alts))
    # plt.plot(alts, Jbcs)
    # plt.plot(alts, Js, '--')
    # plt.yscale("log")
    # plt.plot(alts, Js / Jbcs)
    # plt.plot(alts, Juniform / classical_dilution(alts))
    plt.imshow(grid)
    plt.colorbar()





