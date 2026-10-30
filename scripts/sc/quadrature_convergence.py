"""Angular quadrature convergence of J on a plane-parallel stratified slab
(GivenFs, horizontally homogeneous, periodic x/y, zero incident radiation),
with longer characteristics: exact for cell-constant material along every
direction, so the error against the reference is the quadrature's alone.
Reference: for each mu the exact cell mean of I along the ray,
S + (I_in - S) g(tau) with g = (1 - e^-tau) / tau, integrated over mu with a
dense Gauss-Legendre rule clustered at grazing angles (mu = s^2); mu = 0
(in-plane rays) gives I = S.
Usage: python quadrature_convergence.py DEXRT_SC_3D WORKDIR"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/sc"))
import sc_3d_givenfs_tests as T

binary, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
T.CHARACTERISTICS = "LongerCharacteristics"

nz, n, nw = 64, 8, 16
k = np.arange(nz)
# NOTE: opacity per cell (dz = 1) falling with height; the wavelengths scale
# it so the slab's total optical depth spans ~1e-3 .. 1e4.
shape = np.exp(-(k - nz) / 6.0) / np.exp(nz / 6.0)
scales = np.logspace(-3, 4, nw) / shape.sum()
chi_z = shape[:, None] * scales[None, :]
S_z = (1.0 + 2.0 * (1.0 - k / nz) ** 2)[:, None] * np.ones(nw)[None, :]


def reference():
    s, ws = np.polynomial.legendre.leggauss(4000)
    s = 0.5 * (s + 1.0)
    ws = 0.5 * ws
    mu = s ** 2
    wmu = ws * 2.0 * s
    J = np.zeros((nz, nw))
    for la in range(nw):
        chi, S = chi_z[:, la], S_z[:, la]
        for direction in (1, -1):
            order = k if direction > 0 else k[::-1]
            I_in = np.zeros_like(mu)
            for kk in order:
                tau = chi[kk] / mu
                g = -np.expm1(-tau) / tau
                J[kk, la] += 0.5 * np.sum(wmu * (S[kk] + (I_in - S[kk]) * g))
                I_in = S[kk] + (I_in - S[kk]) * np.exp(-tau)
    return J


J_ref = reference()
eta = np.broadcast_to((chi_z * S_z)[:, None, None, :], (nz, n, n, nw)).astype(np.float32)
chi = np.broadcast_to(chi_z[:, None, None, :], (nz, n, n, nw)).astype(np.float32)
cases = [("Stepan", o) for o in (3, 7, 9, 11, 13, 15)] + [("Healpix", o) for o in (0, 1, 2, 3, 4)]
results = {}
tau_tot = chi_z.sum(axis=0)
groups = (("thin", tau_tot < 0.1), ("mid", (tau_tot >= 0.1) & (tau_tot <= 10)), ("thick", tau_tot > 10))
print("Relative J error, |J - J_ref| / J_ref (median / max) and relative to S, |J - J_ref| / S (max), by slab optical depth")
print(f"{'quadrature':14s} {'N':>5s} " + " ".join(f"{g:>22s}" for g, _ in groups) + f" {'max |dJ|/S':>11s}")
for quad, order in cases:
    out = T.run(binary, work, f"{quad}_{order}", eta, chi, {"quadrature": quad, "quadrature_order": order,
                "x_boundary": "Periodic", "y_boundary": "Periodic"}, [])
    J = np.asarray(out["J"][...]).mean(axis=(2, 3)).T
    nd = out.dimensions["sc_dir"].size
    err = np.abs(J - J_ref) / J_ref
    err_S = np.abs(J - J_ref) / S_z
    results[(quad, order)] = (nd, err, err_S)
    cols = " ".join(f"{np.median(err[:, m]):10.1e} {np.max(err[:, m]):10.1e} " for _, m in groups)
    print(f"{quad + ' ' + str(order):14s} {nd:5d} {cols} {np.max(err_S):11.1e}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for quad, marker in (("Stepan", "o"), ("Healpix", "s")):
    keys = [c for c in cases if c[0] == quad]
    nds = [results[c][0] for c in keys]
    for g, m in groups:
        axes[0].loglog(nds, [np.median(results[c][1][:, m]) for c in keys], marker + "-", label=f"{quad} {g} median")
    axes[0].loglog(nds, [np.max(results[c][2]) for c in keys], marker + "--", label=f"{quad} max |dJ|/S")
axes[0].set_xlabel("directions")
axes[0].set_ylabel("error")
axes[0].legend(fontsize=7)
for c, ls in ((("Stepan", 7), "-"), (("Stepan", 15), "--"), (("Healpix", 2), ":"), (("Healpix", 3), "-.")):
    axes[1].loglog(tau_tot, np.max(results[c][1], axis=0), ls, label=f"{c[0]} {c[1]} ({results[c][0]})")
axes[1].set_xlabel("slab optical depth (per wavelength)")
axes[1].set_ylabel("max over height of relative J error")
axes[1].legend()
fig.tight_layout()
fig.savefig(work / "quadrature_convergence.png", dpi=120)
print("plot:", work / "quadrature_convergence.png")
