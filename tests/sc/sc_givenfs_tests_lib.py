"""Shared helpers for the dexrt_sc GivenFs tests."""
import subprocess
from pathlib import Path
import netCDF4 as ncdf
import yaml


def write_atmos(path, eta, chi, voxel_scale=1.0, dx=None, dz=None, z=None):
    """z: optional non-uniform cell-centre heights (increasing upwards)."""
    nz, nx, nw = eta.shape
    with ncdf.Dataset(path, "w") as f:
        f.createDimension("x", nx)
        f.createDimension("z", nz)
        f.createDimension("wavelength", nw)
        f.createVariable("eta", "f4", ("z", "x", "wavelength"))[...] = eta
        f.createVariable("chi", "f4", ("z", "x", "wavelength"))[...] = chi
        f.createVariable("voxel_scale", "f4")[...] = voxel_scale
        if dx is not None:
            f.createVariable("voxel_scale_x", "f4")[...] = dx
        if dz is not None:
            f.createVariable("voxel_scale_z", "f4")[...] = dz
        if z is not None:
            f.createVariable("z", "f8", ("z",))[...] = z


def run_givenfs(dexrt_sc, work, name, eta, chi, sc, launcher=(), **kw):
    atmos = work / f"{name}_atmos.nc"
    out = work / f"{name}_out.nc"
    write_atmos(atmos, eta, chi, **kw)
    conf = {
        "mode": "GivenFs",
        "atmos_path": str(atmos),
        "output_path": str(out),
        "system": {"mem_pool_gb": 2.0},
        "sc": sc,
    }
    conf_path = work / f"{name}.yaml"
    conf_path.write_text(yaml.dump(conf))
    log_path = work / f"{name}.log"
    with open(log_path, "w") as log:
        res = subprocess.run([*launcher, str(dexrt_sc), "--config", str(conf_path)], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(log_path.read_text()[-3000:])
        raise RuntimeError(f"{name} failed")
    return ncdf.Dataset(out)
