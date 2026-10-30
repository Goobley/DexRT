"""rad_loss: an LTE formal solution in an atmosphere symmetric in z, periodic
in x (and y), with zero inflow, so the volume integral of the net loss equals
the flux leaving through the top and bottom faces (twice the top's, by
symmetry). Short (Linear, MonotonizedCentral) and LongerCharacteristics, 2D
(uniform and a symmetrically stretched z grid) and 3D; and Integrated equals
the wavelength sum of PerWavelength.
Usage: python sc_rad_loss_test.py DEXRT_SC DEXRT_SC_3D WORKDIR"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
from sc_givenfs_tests_lib import emergent_dir_indices

dexrt_sc = Path(sys.argv[1]).resolve()
dexrt_sc_3d = Path(sys.argv[2]).resolve()
work = Path(sys.argv[3]).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
KB = 1.380649e-23
DX = 100e3
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def stretched_z(nz):
    """Cell centres [m], spacing symmetric about the middle (0.5 to 1.5 DX)."""
    steps = DX * (1.0 + 0.5 * np.cos(2 * np.pi * np.arange(1, nz) / nz))
    return np.concatenate([[0.0], np.cumsum(steps)])


def cell_heights(z):
    """Between the midpoint faces, the end faces mirrored."""
    faces = np.concatenate([[z[0] - 0.5 * (z[1] - z[0])], 0.5 * (z[1:] + z[:-1]), [z[-1] + 0.5 * (z[-1] - z[-2])]])
    return np.diff(faces)


def write_atmos(path, shape, z=None):
    """Symmetric in z about the middle; a hot blob and a dense one."""
    axes = [(np.arange(n) + 0.5) / n for n in shape]
    if z is not None:
        axes[0] = (z - z[0] + 0.5 * (z[1] - z[0])) / (z[-1] - z[0] + 0.5 * (z[1] - z[0]) + 0.5 * (z[-1] - z[-2]))
    grids = np.meshgrid(*axes, indexing="ij")
    Z, X = grids[0], grids[-1]
    Y = grids[1] if len(shape) == 3 else 0.5
    T = 6000.0 + 4000.0 * np.exp(-((X - 0.4) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / 0.02)
    nh = 3e16 * (1.0 + 2.0 * np.exp(-((X - 0.7) ** 2 + (Z - 0.5) ** 2) / 0.03))
    dims = ("z", "y", "x")[-len(shape):] if len(shape) == 3 else ("z", "x")
    with ncdf.Dataset(path, "w") as f:
        for d, n in zip(dims, shape):
            f.createDimension(d, n)
        for k, v in dict(temperature=T, ne=0.2 * nh, nh_tot=nh, vturb=0 * T + 5e3, pressure=nh * KB * T,
                         vx=0 * T, vy=0 * T, vz=0 * T).items():
            f.createVariable(k, "f4", dims)[...] = v
        f.createVariable("voxel_scale", "f4")[...] = DX
        if z is not None:
            f.createVariable("z", "f8", ("z",))[...] = z


def run(exe, name, atmos, sc, rad_loss):
    conf = {
        "mode": "Lte", "atmos_path": str(atmos), "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_4.yaml"), "treatment": "Active"}},
        "boundary_type": "Zero", "rad_loss": rad_loss, "system": {"mem_pool_gb": 1.0},
        "output": {"rad_loss": True, "wavelength": True, "sc_emergent": True},
        "sc": {"quadrature": "Stepan", "quadrature_order": 3, "x_boundary": "Periodic", "y_boundary": "Periodic", **sc},
    }
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    with open(work / f"{name}.log", "w") as log:
        res = subprocess.run([str(exe), "--config", str(work / f"{name}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print((work / f"{name}.log").read_text()[-3000:])
        raise RuntimeError(name)
    return ncdf.Dataset(work / f"{name}.nc")


def energy(ds, three_d, z=None):
    """(volume integral of the loss, flux out of both faces), per wavelength."""
    L = np.asarray(ds["rad_loss"][...], dtype=np.float64)
    H = cell_heights(z) if z is not None else np.full(L.shape[1], DX)
    V = H.reshape(-1, *([1] * (L.ndim - 2))) * DX ** (2 if three_d else 1)
    loss = (L * V).reshape(L.shape[0], -1).sum(axis=1)
    indices = emergent_dir_indices(ds)
    mu = np.asarray(ds["sc_mu"][...])[indices]
    wmu = np.asarray(ds["sc_wmu"][...])[indices]
    I = np.asarray(ds["I_emergent"][...], dtype=np.float64)
    I = I.reshape(I.shape[0], I.shape[1], -1)
    up = mu[:, 2] > 0
    face = DX ** (2 if three_d else 1)
    flux = 4 * np.pi * np.einsum("d,ldk->l", (wmu * np.abs(mu[:, 2]))[up], I[:, up]) * face
    return loss, 2.0 * flux * np.asarray(ds["wavelength_bin"][...])


cases = [("Short", "Linear"), ("Short", "MonotonizedCentral"), ("LongerCharacteristics", "Linear")]
for three_d, exe, shape, z in [(False, dexrt_sc, (64, 32), None),
                               (False, dexrt_sc, (64, 32), stretched_z(64)),
                               (True, dexrt_sc_3d, (32, 16, 16), None)]:
    tag = ("3d" if three_d else "2d") + ("_z" if z is not None else "")
    atmos = work / f"atmos_{tag}.nc"
    write_atmos(atmos, shape, z)
    for chars, interp in cases:
        name = f"{tag}_{chars}_{interp}"
        sc = {"characteristics": chars, "interpolation": interp}
        per = run(exe, name + "_per", atmos, sc, "PerWavelength")
        integ = run(exe, name + "_int", atmos, sc, "Integrated")
        loss, flux = energy(per, three_d, z)
        err = np.max(np.abs(loss - flux)) / np.max(np.abs(flux))
        check(err < 1e-4, f"{name}: loss volume integral = outgoing flux per wavelength (max err {err:.2e} of the peak)")
        total = np.abs(loss.sum() - flux.sum()) / flux.sum()
        check(total < 1e-5, f"{name}: total loss = total outgoing flux (rel err {total:.2e})")
        Lp = np.asarray(per["rad_loss"][...], dtype=np.float64).sum(axis=0)
        Li = np.asarray(integ["rad_loss"][0], dtype=np.float64)
        rel = np.max(np.abs(Lp - Li)) / np.max(np.abs(Li))
        check(rel < 1e-5, f"{name}: Integrated = sum of PerWavelength (max diff {rel:.2e} of the peak)")

if failures:
    print(f"{len(failures)} failures")
    sys.exit(1)
print("All rad_loss tests passed")
