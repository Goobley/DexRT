"""Ingredient-level H_6 comparison with Lightweaver on the lw_compare atmosphere
(fixed zero boundaries): LTE populations, collisional rates, and radiative
rates from one formal solution started from DexRT's LTE populations.
Usage: python lw_compare_rates.py WORKDIR  (needs rates_sc_out.nc from dexrt_sc)"""
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom
sys.path.insert(0, str(Path(__file__).parent))
from lw_models import crtaf_h6

work = Path(sys.argv[1]).resolve()
nx, nz, dx = 64, 64, 100e3
zz, xx = np.meshgrid((np.arange(nz) + 0.5) / nz, (np.arange(nx) + 0.5) / nx, indexing="ij")
T = 6000.0 + 4000.0 * np.exp(-((xx - 0.4) ** 2 + (zz - 0.55) ** 2) / 0.02) + 1500.0 * zz
nh = 3e16 * (1.0 + 2.0 * np.exp(-((xx - 0.6) ** 2 + (zz - 0.3) ** 2) / 0.03))
ne = 0.2 * nh
vturb = np.full_like(T, 5e3)


class ZeroXBc(lw.BoundaryCondition):
    def compute_bc(self, atmos, spect):
        return np.zeros((spect.wavelength.shape[0], self.mux.shape[0], atmos.Nz))


def pad(a, scale=1.0):
    a = np.concatenate([a[:1] * scale, a, a[-1:] * scale], axis=0)
    return np.concatenate([a[:, :1] * scale, a, a[:, -1:] * scale], axis=1)


flip = lambda a: np.ascontiguousarray(a[::-1].astype(np.float64))
unpad = lambda a: a.reshape(a.shape[:-1] + (nz + 2, nx + 2))[..., ::-1, :][..., 1:-1, 1:-1]


def to_lw(a):
    """[..., z, x] dexrt -> [..., Nspace] lightweaver (padded, top first)."""
    out = np.stack([flip(pad(a[i])) for i in range(a.shape[0])]).reshape(a.shape[0], -1)
    return out


atmos = lw.Atmosphere.make_2d(
    height=((np.arange(nz + 2) - 0.5) * dx)[::-1].copy(), x=(np.arange(nx + 2) - 0.5) * dx,
    temperature=flip(pad(T)), vx=flip(pad(0 * T)), vz=flip(pad(0 * T)), vturb=flip(pad(vturb)),
    ne=flip(pad(ne, 1e-12)), nh_tot=flip(pad(nh, 1e-12)),
    x_lower_bc=ZeroXBc(), x_upper_bc=ZeroXBc(),
    z_lower_bc=lw.ZeroRadiation(), z_upper_bc=lw.ZeroRadiation(),
)
atmos.quadrature(3)
h_model = crtaf_h6() if "--crtaf" in sys.argv else H_6_atom()
print("Lightweaver H model:", "CRTAF tests/H_6.yaml" if "--crtaf" in sys.argv else "rh_atoms.H_6_atom")
a_set = lw.RadiativeSet([h_model])
a_set.set_active("H")
spect = a_set.compute_wavelength_grid()
eq_pops = a_set.compute_eq_pops(atmos)

dex = ncdf.Dataset(work / "rates_sc_out.nc")
dex_lte = np.asarray(dex["lte_pops"][...])[:, :nz, :nx].astype(np.float64)
dex_C = np.asarray(dex["collisional_rates_0"][...])[..., :nz, :nx]
dex_R = np.asarray(dex["radiative_rates_0"][...])[..., :nz, :nx]

H = eq_pops.atomic_pops["H"] if hasattr(eq_pops, "atomic_pops") else eq_pops.atomicPops["H"]
lw_nstar = unpad(np.asarray(H.n_star).copy())
print("== LTE populations (Lightweaver incl. Debye) vs DexRT")
for l in range(6):
    r = lw_nstar[l] / dex_lte[l] - 1
    print(f"  level {l}: median {np.median(r):+.3e}  range [{r.min():+.2e}, {r.max():+.2e}]")

# Start Lightweaver from DexRT's LTE populations (both n and n_star).
lte_lw = to_lw(dex_lte)
pad_rows = np.asarray(H.n_star).reshape(6, nz + 2, nx + 2)
np.asarray(H.n)[:] = lte_lw
np.asarray(H.n_star)[:] = lte_lw
ctx = lw.Context(atmos, spect, eq_pops, Nthreads=16, formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
ctx.background.eta[...] = 0.0
ctx.background.chi[...] = 0.0
ctx.background.sca[...] = 0.0
np.asarray(H.n)[:] = lte_lw
np.asarray(H.n_star)[:] = lte_lw
ctx.formal_sol_gamma_matrices(lambda_iterate=True)

atom = ctx.active_atoms[0] if hasattr(ctx, "active_atoms") else ctx.activeAtoms[0]
lw_C = unpad(np.asarray(atom.C).copy())
print("== Collisional rates C[to, from] (Lightweaver / DexRT - 1), median over space")
for i in range(6):
    for j in range(6):
        if i == j:
            continue
        a, b = lw_C[i, j], dex_C[i, j]
        if np.all(b == 0) and np.all(a == 0):
            continue
        r = np.median(a / np.where(b != 0, b, np.nan) - 1)
        print(f"  {j}->{i}: {r:+.3e}   (LW {np.median(a):.3e}, DexRT {np.median(b):.3e})")

print("== Radiative rates (Lightweaver / DexRT - 1), median over space")
for t in atom.trans:
    i, j = t.i, t.j
    Rij = unpad(np.asarray(t.Rij).copy())
    Rji = unpad(np.asarray(t.Rji).copy())
    up = np.median(Rij / dex_R[j, i] - 1)
    down = np.median(Rji / dex_R[i, j] - 1)
    print(f"  {i}<->{j}: up {up:+.3e}, down {down:+.3e}   (LW up {np.median(Rij):.3e}, down {np.median(Rji):.3e})")
