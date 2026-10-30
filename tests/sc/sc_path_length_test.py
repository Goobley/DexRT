"""Uniform slab, periodic in x, Zero z boundaries: I = S (1 - exp(-chi L))
exactly for every node and direction (L the distance back to the z boundary;
I = S for horizontal rays), and interpolation along rows is exact. Checks the
characteristic geometry (row steps, multi-column crossings, edge starts).
With fixed x the side kink in I(x) is smeared by the interpolation, so that
case is only reported.
Usage: python sc_path_length_test.py DEXRT_SC WORKDIR"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import run_givenfs

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
nx, nz = 64, 32
# NOTE: J is stored in 16 bits (2^-11 relative), re-rounded per direction batch.
RAD_RTOL = 2e-3
failures = []
for (dx, dz) in [(1.0, 1.0), (1.0, 2.5)]:
    for chi_val in [1e-3, 0.3]:
        chi = np.full((nz, nx, 1), chi_val, dtype=np.float32)
        eta = chi * 2.0
        for (quad, order, periodic) in [(q, o, per) for q, o in [("Stepan", 7), ("Healpix", 2)] for per in [True, False]]:
            sc = {"characteristics": "Short", "quadrature": quad, "quadrature_order": order, "formal_solver": "PiecewiseLinear", "interpolation": "Linear",
                  "x_boundary": "Periodic" if periodic else "Fixed"}
            out = run_givenfs(dexrt_sc, work, f"path_{dx}_{dz}_{chi_val}_{quad}_{periodic}", eta, chi, sc, dx=dx, dz=dz)
            mu = np.asarray(out["sc_mu"][...])
            wmu = np.asarray(out["sc_wmu"][...])
            J = np.asarray(out["J"][0])[:nz, :nx]
            X, Z = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(nz) + 0.5) * dz)
            J_exact = np.zeros_like(X)
            for d in range(mu.shape[0]):
                mx, mz = mu[d, 0], mu[d, 2]
                L = np.full_like(X, np.inf)
                if mx > 0 and not periodic: L = np.minimum(L, X / mx)
                if mx < 0 and not periodic: L = np.minimum(L, (nx * dx - X) / -mx)
                if mz > 0: L = np.minimum(L, Z / mz)
                if mz < 0: L = np.minimum(L, (nz * dz - Z) / -mz)
                J_exact += wmu[d] * 2.0 * -np.expm1(-chi_val * L)
            err = np.abs(J / J_exact - 1)
            msg = f"dx={dx} dz={dz} chi={chi_val} {quad} {'periodic' if periodic else 'fixed'}: max rel err {err.max():.2e}, mean {err.mean():.2e}"
            print(msg)
            if periodic and err.max() > RAD_RTOL:
                failures.append(msg)
for f in failures:
    print("FAIL", f)
sys.exit(1 if failures else 0)
