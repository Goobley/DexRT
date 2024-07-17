import crtaf
from crtaf.from_lightweaver import LightweaverAtomConverter
from crtaf.core_types import TemperatureInterpolationRateImpl, CERate, CIRate
from lightweaver.rh_atoms import H_6_atom, CaII_atom, H_4_atom
import numpy as np
import astropy.units as u
import astropy.constants as const

h_coll_temperature_grid = np.array([3e3, 5e3, 7e3, 10e3, 20e3, 30e3, 50e3, 100e3, 1e6, 2e6])

# NOTE(cmo): Using RH implementation as the lower bound in Johnson is defined differently to scipy
# RH implementation based on  Handbook of Mathematical Functions, Applied mathematics series 55 (1964) (ed. Abramovitz and Stegun).
def E1(x):
    a53 = np.array([
        -0.57721566,
        0.99999193,
        -0.24991055,
        0.05519968,
        -0.00976004,
        0.00107857
    ])
    a56 = np.array([
        8.5733287401,
        18.0590169730,
        8.6347608925,
        0.2677737343
    ])
    b56 = np.array([
        9.5733223454,
        25.6329561486,
        21.0996530827,
        3.9584969228
    ])

    if np.any(x <= 0.0):
        raise ValueError("x < 0")

    mask = (x <= 1.0)
    E1 = np.zeros_like(x)
    xm = x[mask]
    # E1[mask] = -np.log(xm) + a53[0] + xm * a53[1] + xm**2 * a53[2] + xm**3 * a53[3] + xm**4 * a53[4] + xm**5 * a53[5]
    E1[mask] = -np.log(xm) + a53[0] + xm*(a53[1] + xm*(a53[2] + xm*(a53[3] + xm*(a53[4] + xm *a53[5]))))

    xm = x[~mask]
    E1[~mask] = a56[3] / xm + a56[2] + xm * (a56[1] + xm * (a56[0] + xm))
    E1[~mask] /= b56[3] + xm * (b56[2] + xm * (b56[1] + xm * (b56[0] + xm)))
    E1[~mask] *= np.exp(-xm)
    return E1

def E2(x):
    return np.exp(-x) - x * E1(x)

def expn(n, i):
    if n == 1:
        return E1(i)
    if n == 2:
        return E2(i)
    raise ValueError("Invalid exp integral order requested")

def fn_f(x):
    return np.exp(-x) / x - 2.0 * E1(x) + E2(x)

def g0(n):
    if n == 1:
        return 1.133
    elif n == 2:
        return 1.0785
    return 0.9935 + (0.2328 - 0.1296/n)/n

def g1(n):
    if n == 1:
        return -0.4059
    elif n == 2:
        return -0.2319
    return -(0.6282 - (0.5598 - 0.5299/n)/n) / n

def g2(n):
    if n == 1:
        return  0.07014
    elif n == 2:
        return  0.02947
    return (0.3887 - (1.181 - 1.4700/n)/n) / (n*n)

def Johnson_CE(i, j, Eij, Te):
    y = Eij / (const.k_B.value * Te)
    C1 = 32.0 / (3.0 * np.sqrt(3.0) * np.pi)

    if i == 1:
        ri = 0.45
        bi = -0.603
    else:
        ri = 1.94 * i**(-1.57)
        bi = (4.0 + (-18.63 + (36.24 - 28.09 / i) / i) / i) / i
    xr = 1.0 - i**2 / j**2
    # NOTE(cmo): rij is calculated incorrectly in rh's make_h.c for higher order
    # CE terms in a series, due to the incorrect *= accumulation
    rij = ri * xr
    z = rij + y

    fij = C1 * i / (j * xr)**3 * (g0(i) + (g1(i) + g2(i) / xr) / xr)
    Aij = 2.0 * i**2 / xr * fij
    Bij = 4.0 * i**4 / (j**3 * xr**2) * (1.0 + 4.0 / (3.0 * xr) + bi / xr**2)
    ce = np.sqrt((8.0 * const.k_B.value * Te) / (np.pi * const.m_e.value)) * (2.0 * np.pi * const.a0.value**2 * y**2 * i**2) / xr
    t1 = Aij * ((0.5 + 1.0 / y) * expn(1, y) - (0.5 + 1.0 / z) * expn(1, z))
    t2 = (Bij - Aij * np.log(2*i**2 / xr)) * (expn(2, y) / y - expn(2, z) / z)
    ce *= (t1 + t2)
    ce *= np.exp(y) / np.sqrt(Te)
    return ce

def Johnson_CI(i, Eij, Te):
    y = Eij / (const.k_B.value * Te)
    C1 = 32.0 / (3.0 * np.sqrt(3.0) * np.pi)

    if i == 1:
        ri = 0.45
        bi = -0.603
    else:
        ri = 1.94 * i**(-1.57)
        bi = (4.0 + (-18.63 + (36.24 - 28.09 / i) / i) / i) / i
    z = ri + y

    An = C1 * i * (g0(i) / 3.0 + g1(i) / 4.0 + g2(i) / 5.0)
    Bn = 2.0 * i**2 / 3.0 * (5.0 + bi)
    ci = np.sqrt((8.0 * const.k_B.value * Te) / (np.pi * const.m_e.value)) * (2.0 * np.pi * const.a0.value**2 * y**2 * i**2)
    t1 = An * (E1(y) / y - E1(z) / z)
    t2 = (Bn - An * np.log(2.0 * i**2)) * (fn_f(y) - fn_f(z))
    ci *= (t1 + t2)
    ci *= np.exp(y) / np.sqrt(Te)
    return ci


def make_atom():
    conv = LightweaverAtomConverter()
    model = conv.convert(H_6_atom())
    for l in model.lines:
        l.wavelength_grid.q_core *= 4
        l.wavelength_grid.q_wing *= 5
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)
    for coll_trans in model_simplified.collisions:
        for coll in coll_trans.data:
            if isinstance(coll, (CERate, CIRate)):
                coll.temperature = h_coll_temperature_grid << u.K
                rate_unit = coll.data.unit
                Eij = (model_simplified.levels[coll_trans.transition[0]].energy_eV - model_simplified.levels[coll_trans.transition[1]].energy_eV).to(u.J)
                n = np.sqrt(model_simplified.levels[coll_trans.transition[1]].g / 2)
                if isinstance(coll, CERate):
                    nn = np.sqrt(model_simplified.levels[coll_trans.transition[0]].g / 2)
                    coll.data = Johnson_CE(n, nn, Eij.value, h_coll_temperature_grid) << rate_unit
                elif isinstance(coll, CIRate):
                    coll.data = Johnson_CI(n, Eij.value, h_coll_temperature_grid) << rate_unit
    # new_lines = []
    # for l in model_simplified.lines:
    #     if l.lambda0 < 1000.0 * u.nm:
    #         new_lines.append(l)
    # model_simplified.lines = new_lines
    return model_simplified

def make_H_4():
    conv = LightweaverAtomConverter()
    model = conv.convert(H_4_atom())
    for l in model.lines:
        l.wavelength_grid.q_core *= 3
        l.wavelength_grid.q_wing *= 2
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)
    return model_simplified

def make_CaII():
    conv = LightweaverAtomConverter()
    model = conv.convert(CaII_atom())
    for l in model.lines:
        l.wavelength_grid.q_core *= 3
        l.wavelength_grid.q_wing *= 2
        # l.wavelength_grid.n_lambda //= 2
    visitor = crtaf.AtomicSimplificationVisitor(crtaf.default_visitors())
    model_simplified = model.simplify_visit(visitor)

    current_grid = model_simplified.lines[-1].wavelength_grid.wavelengths
    new_grid = np.sort(np.concatenate((current_grid, [-1.0 * u.nm])))
    model_simplified.lines[-1].wavelength_grid.wavelengths = new_grid

    # NOTE(cmo): To prevent explosion due to rates in the Snow KHI model
    # TODO(cmo): Grab the rates from source/RADYN
    for trans in model_simplified.collisions:
        for coll in trans.data:
            if isinstance(coll, TemperatureInterpolationRateImpl) and coll.temperature[0] > (1000.0 * u.K):
                coll.temperature = np.concatenate(([500.0 * u.K], coll.temperature))
                coll.data = np.concatenate(([0.0 * coll.data.unit], coll.data))
    return model_simplified


if __name__ == "__main__":
    atom = make_CaII()
    with open("test_CaII.yaml", "w") as f:
        f.write(atom.yaml_dumps())

    atom = make_H_4()
    with open("H_4.yaml", "w") as f:
        f.write(atom.yaml_dumps())

    atom = make_atom()
    with open("H_6.yaml", "w") as f:
        f.write(atom.yaml_dumps())