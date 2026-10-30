"""A y-invariant 3D problem (periodic y) against the 2D solver, GivenFs.
Longer characteristics agree to rounding (y crossings only split
constant-material segments); short characteristics to truncation level
(their extra crossings split the formal steps).
Usage: python sc_3d_vs_2d.py DEXRT_SC DEXRT_SC_3D WORKDIR"""
import sys
from pathlib import Path
import numpy as np
import subprocess
import netCDF4 as ncdf
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from sc_givenfs_tests_lib import write_atmos as write_atmos_2d
from sc_3d_givenfs_tests import write_atmos as write_atmos_3d


def run(binary, work, name, sc, writer, eta, chi):
    atmos = work / f"{name}_atmos.nc"
    out = work / f"{name}_out.nc"
    writer(atmos, eta, chi)
    conf = {"mode": "GivenFs", "atmos_path": str(atmos), "output_path": str(out),
            "system": {"mem_pool_gb": 2.0}, "sc": sc}
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    with open(work / f"{name}.log", "w") as log:
        subprocess.run([str(binary), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    return np.asarray(ncdf.Dataset(out)["J"][...])


def main():
    sc2, sc3, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    nz, nx, ny, nw = 32, 48, 8, 3
    z, x = np.mgrid[0:nz, 0:nx].astype(np.float64)
    blob = np.exp(-((x - 20) ** 2 / 30 + (z - 12) ** 2 / 20))
    chi = np.stack([(0.02 + k * 0.3) * (1 + 3 * blob) for k in range(nw)], axis=-1).astype(np.float32)
    eta = (chi * (0.5 + 5 * blob[..., None])).astype(np.float32)
    eta3 = np.repeat(eta[:, None], ny, axis=1)
    chi3 = np.repeat(chi[:, None], ny, axis=1)
    worst = {}
    for ch, interp, quad, tol in (("LongerCharacteristics", "Linear", "Stepan", 1e-5),
                                  ("Short", "Linear", "Stepan", 3e-2),
                                  ("Short", "MonotonizedCentral", "Stepan", 3e-2)):
        for xb in ("Fixed", "Periodic"):
            sc = {"characteristics": ch, "interpolation": interp, "formal_solver": "PiecewiseLinear", "x_boundary": xb,
                  "quadrature": quad, "quadrature_order": 7 if quad == "Stepan" else 2}
            name = f"{ch}_{interp}_{quad}_{xb}"
            J2 = run(sc2, work, name + "_2d", sc, write_atmos_2d, eta, chi)
            J3 = run(sc3, work, name + "_3d", {**sc, "y_boundary": "Periodic"}, write_atmos_3d, eta3, chi3)
            spread_y = np.max(np.abs(J3 - J3[:, :, :1]) / np.max(np.abs(J3)))
            err = np.max(np.abs(J3 - J2[:, :, None, :]) / np.max(np.abs(J2)))
            med = np.median(np.abs(J3 - J2[:, :, None, :]) / np.maximum(np.abs(J2[:, :, None, :]), 1e-30))
            print(f"{name}: max |J3d - J2d| / max {err:.2e}, median rel {med:.2e}, y variation {spread_y:.1e}")
            assert spread_y < 1e-5, spread_y
            assert err < tol, err
    print("3D vs 2D: passed")


if __name__ == "__main__":
    main()
