import yaml
import numpy as np

# theta = np.linspace(0.0, 90.0, 91)
theta = -90.0
muz = np.cos(np.deg2rad(theta))
mux = np.sign(theta) * np.sqrt(1.0 - muz**2)
lambda0 = 393.47771342231175
# lambda0 = 854.4437912696352
delta_lambda = 0.1
# lambda0 = 121.56841096386113
# lambda0 = 102.57334047695103
# lambda0 = 656.4691622298104
# delta_lambda = 0.2
wavelengths = np.linspace(lambda0 - delta_lambda, lambda0 + delta_lambda, 251)

path = "valeriia_a0500.yaml"
filename = f"dexrt_ray_{path[:-5]}_cfn_prom.yaml"
data = {
    "dexrt_config_path": path,
    "ray_output_path": path[:-5] + f"_{int(lambda0)}_cfn_prom.nc",

    # "atmos_path": "snow_atmos_steeper_10Mm.nc",
    # "dex_output_path": "output_snow_khi_10Mm_H3CaFull_v6_mhdvel.nc",
    "rotate_aabb": True,
    "muz": muz.tolist(),
    "mux": mux.tolist(),
    "wavelength": wavelengths.tolist(),
    "output_cfn": False,
    "output_eta_chi": True,
    "system": {
        "mem_pool_initial_gb": 6.0,
        "mem_pool_grow_gb": 2.0,
    }
}

if __name__ == "__main__":
    with open(filename, "w") as f:
        yaml.dump(data, f)