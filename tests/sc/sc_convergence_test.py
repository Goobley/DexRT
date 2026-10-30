"""Resolution study: smooth, periodic-in-x, optically thin/intermediate field.
Grids refined by 3x share node positions (cell centres), so J on the coarse
nodes can be compared directly to the finest solution.
Usage: python sc_convergence_test.py DEXRT_SC WORKDIR"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import run_givenfs

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
L = 1.0
n0 = 64
refine = [1, 3, 9]


def field(n):
    c = (np.arange(n) + 0.5) / n
    Z, X = np.meshgrid(c, c, indexing="ij")
    blob = np.exp(-((X - 0.4) ** 2 + (Z - 0.5) ** 2) / 0.01) + 0.5 * np.exp(-((X - 0.75) ** 2 + (Z - 0.3) ** 2) / 0.004)
    chi = np.stack([0.02 * (0.1 + blob), 2.0 * (0.1 + blob)], axis=-1)  # per unit length: thin, intermediate
    S = (1.0 + 3.0 * blob)[..., None]
    return chi, chi * S


for interp in ["Linear", "MonotonizedCentral"]:
    sols = {}
    for r in refine:
        n = n0 * r
        chi, eta = field(n)
        sc = {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
              "interpolation": interp, "x_boundary": "Periodic"}
        out = run_givenfs(dexrt_sc, work, f"conv_{interp}_{n}", (eta * L / n).astype(np.float32) * n / L,
                          (chi * L / n).astype(np.float32) * n / L, sc, voxel_scale=L / n)
        J = np.asarray(out["J"][...])[:, :n, :n]
        sols[r] = J[:, r // 2::r, r // 2::r]
    ref = sols[refine[-1]]
    for r in refine[:-1]:
        for la, name in enumerate(["thin", "intermediate"]):
            e = (sols[r][la] - ref[la]) / ref[la]
            print(f"{interp:20s} {name:12s} n={n0 * r:4d}: mean signed {e.mean():+.2e}, mean |e| {np.abs(e).mean():.2e}, max {np.abs(e).max():.2e}")
