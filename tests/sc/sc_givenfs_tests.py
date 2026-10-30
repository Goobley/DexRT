"""GivenFs tests for dexrt_sc: searchlight beam, optically thick slab, and
periodic shift invariance. Usage: python sc_givenfs_tests.py DEXRT_SC WORKDIR"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import run_givenfs

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)


failures = []
# NOTE: J is stored in 16 bits (2^-11 relative), re-rounded per direction batch.
RAD_RTOL = 2e-3


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# Optically thick homogeneous medium: J -> S in the interior.
n = 64
S_val = 2.5
chi = np.full((n, n, 1), 10.0, dtype=np.float32)
eta = chi * S_val
for solver in ["PiecewiseConstant", "PiecewiseLinear"]:
    for interp in ["Linear", "VanLeer", "MonotonizedCentral"]:
        sc = {"characteristics": "Short", "formal_solver": solver, "interpolation": interp, "quadrature": "Stepan", "quadrature_order": 7}
        out = run_givenfs(dexrt_sc, work, f"thick_{solver}_{interp}", eta, chi, sc)
        J = np.asarray(out["J"][0, :n, :n])
        interior = J[16:-16, 16:-16]
        check(np.allclose(interior, S_val, rtol=RAD_RTOL), f"thick {solver}/{interp}: interior J = S (max err {np.max(np.abs(interior / S_val - 1)):.2e})")
        # Edge: J at the boundary cell approaches S/2-ish, and is below S
        check(J[0, n // 2] < S_val * (1 + RAD_RTOL), f"thick {solver}/{interp}: boundary J <= S")

# Optically thick on a stretched z grid (geometric, then a 3x jump).
z = np.cumsum(np.where(np.arange(n) < n // 2, 1.04 ** np.arange(n), 3.0 * 1.04 ** (n // 2)))
chi = np.full((n, n, 1), 10.0, dtype=np.float32)
eta = chi * S_val
for interp in ["Linear", "MonotonizedCentral"]:
    sc = {"characteristics": "Short", "formal_solver": "PiecewiseLinear", "interpolation": interp, "quadrature": "Stepan", "quadrature_order": 7}
    out = run_givenfs(dexrt_sc, work, f"thick_nonuniform_{interp}", eta, chi, sc, z=z)
    J = np.asarray(out["J"][0, :n, :n])
    interior = J[16:-16, 16:-16]
    check(np.allclose(interior, S_val, rtol=RAD_RTOL), f"thick non-uniform z {interp}: interior J = S (max err {np.max(np.abs(interior / S_val - 1)):.2e})")
    check(np.allclose(np.asarray(out["z"][:]), z), f"thick non-uniform z {interp}: z written to output")

# Longer characteristics: thick medium J -> S (uniform and stretched z).
chi = np.full((n, n, 1), 10.0, dtype=np.float32)
eta = chi * S_val
for zt, zz in [("uniform", None), ("stretched", z)]:
    sc = {"characteristics": "LongerCharacteristics", "quadrature": "Stepan", "quadrature_order": 7}
    out = run_givenfs(dexrt_sc, work, f"thick_longer_{zt}", eta, chi, sc, z=zz)
    J = np.asarray(out["J"][0, :n, :n])
    interior = J[16:-16, 16:-16]
    check(np.allclose(interior, S_val, rtol=RAD_RTOL), f"thick longer {zt}: interior J = S (max err {np.max(np.abs(interior / S_val - 1)):.2e})")
    check(J[0, n // 2] < S_val * (1 + RAD_RTOL), f"thick longer {zt}: boundary J <= S")

# Uniform emitting slab, thin: J at the centre of a uniform layer of finite
# optical depth must be <= S and symmetric.
chi = np.full((n, n, 1), 0.05, dtype=np.float32)
eta = chi * 1.0
out = run_givenfs(dexrt_sc, work, "thin_slab", eta, chi, {"characteristics": "Short", "formal_solver": "PiecewiseLinear", "interpolation": "Linear"})
J = np.asarray(out["J"][0, :n, :n])
check(np.allclose(J, J[::-1, :], rtol=RAD_RTOL) and np.allclose(J, J[:, ::-1], rtol=RAD_RTOL), "thin slab: J symmetric")
check(np.all(J > 0) and np.all(J < 1.0), "thin slab: 0 < J < S")

# Periodic: a source pattern shifted in x gives shifted J.
nx, nz = 64, 64
xx = np.arange(nx)
zz = np.arange(nz)
Z, X = np.meshgrid(zz, xx, indexing="ij")
chi = (0.02 + 0.5 * np.exp(-((X - 20) ** 2 + (Z - 24) ** 2) / 20.0)).astype(np.float32)[..., None]
eta = chi * (1.0 + np.exp(-((X - 20) ** 2 + (Z - 24) ** 2) / 20.0))[..., None]
sc = {"characteristics": "Short", "formal_solver": "PiecewiseLinear", "interpolation": "MonotonizedCentral", "x_boundary": "Periodic"}
a = run_givenfs(dexrt_sc, work, "periodic_a", eta, chi, sc)
shift = 17
b = run_givenfs(dexrt_sc, work, "periodic_b", np.roll(eta, shift, axis=1), np.roll(chi, shift, axis=1), sc)
Ja = np.asarray(a["J"][0, :nz, :nx])
Jb = np.asarray(b["J"][0, :nz, :nx])
err = np.max(np.abs(np.roll(Ja, shift, axis=1) - Jb) / np.abs(Jb))
check(err < 1e-3, f"periodic: J shift invariant (max rel err {err:.2e})")
sc_l = {"characteristics": "LongerCharacteristics", "x_boundary": "Periodic"}
a = run_givenfs(dexrt_sc, work, "periodic_longer_a", eta, chi, sc_l)
b = run_givenfs(dexrt_sc, work, "periodic_longer_b", np.roll(eta, shift, axis=1), np.roll(chi, shift, axis=1), sc_l)
Ja = np.asarray(a["J"][0, :nz, :nx])
Jb = np.asarray(b["J"][0, :nz, :nx])
err = np.max(np.abs(np.roll(Ja, shift, axis=1) - Jb) / np.abs(Jb))
check(err < 1e-5, f"periodic longer: J shift invariant (max rel err {err:.2e})")

# Very narrow periodic cells force >32 crossings per row (and several domain
# laps) for the oblique Stepan rays. Longer must still give the exact cell mean
# of a plane-parallel slab, including when segment processing is blocked.
nx, nz = 16, 8
for chi_val in [1e-3, 0.3]:
    chi = np.full((nz, nx, 1), chi_val, dtype=np.float32)
    out = run_givenfs(dexrt_sc, work, f"blocked_longer_{chi_val}", chi * 2.0, chi,
                     {"characteristics": "LongerCharacteristics", "quadrature_order": 7,
                      "x_boundary": "Periodic"}, dx=0.02, dz=1.0)
    mu, weights = np.asarray(out["sc_mu"][:]), np.asarray(out["sc_wmu"][:])
    expected = np.zeros(nz)
    for direction, weight in zip(mu, weights):
        az = abs(direction[2])
        tau = chi_val / az
        entry = (np.arange(nz) if direction[2] > 0 else np.arange(nz)[::-1]) / az
        mean_attenuation = -np.expm1(-tau) / tau
        expected += weight * 2.0 * (1.0 - np.exp(-chi_val * entry) * mean_attenuation)
    J = np.asarray(out["J"][0])
    err = np.max(np.abs(J / expected[:, None] - 1.0))
    check(err < RAD_RTOL, f"blocked longer chi={chi_val}: analytic cell mean (max rel err {err:.2e})")

if failures:
    print(f"{len(failures)} failures")
    sys.exit(1)
print("All GivenFs tests passed")
