from dexrt_py.dexrt.config_schemas.dexrt_ray import DexrtRayConfig
from dexrt_py.dexrt.write_config import write_config
import xarray as xr
import numpy as np

if __name__ == "__main__":
    prefix = "jk20200550"
    dex_out = xr.open_dataset(f"JackHighRes/{prefix}_synth.nc")
    lambda0s = dex_out.attrs["lambda0"]
    delta_lambda = 0.2
    wave_grids = [
        np.linspace(l-delta_lambda, l+delta_lambda, 251) for l in lambda0s
    ]
    theta = np.linspace(-90.0, 0.0, 91)
    muz = np.cos(np.deg2rad(theta))
    mux = np.sign(theta) * np.sqrt(1.0 - muz**2)

    config = DexrtRayConfig(
        dexrt_config_path=f"{prefix}.yaml",
        ray_output_path=f"{prefix}_ray.nc",
        wavelength=np.sort(np.concatenate(wave_grids)).tolist(),
        muz=muz.tolist(),
        mux=mux.tolist(),
    )

    write_config(config, f"JackHighRes/{prefix}_ray.yaml")