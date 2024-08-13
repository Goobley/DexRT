import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import yaml
from scipy.ndimage import rotate, convolve1d
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from weno4 import weno4
import astropy.constants as const
import astropy.units as u
from numba import njit
import promweaver as pw
import xarray as xr

prefix = "JackHighRes"
file_id = "jk20200550"
config_file = f"{file_id}_ray.yaml"

# 10 Mm table
temperature_grid = np.array([6e3, 8e3, 10e3, 12e3, 14e3])
pressure_grid = np.array([0.01, 0.02, 0.05, 0.10, 0.20]) * 0.1 # SI
ionisation_table = np.array([
    [0.74, 0.62, 0.44, 0.31, 0.20],
    [0.83, 0.72, 0.55, 0.44, 0.35],
    [0.87, 0.79, 0.70, 0.69, 0.73],
    [0.91, 0.85, 0.82, 0.85, 0.89],
    [0.93, 0.89, 0.89, 0.92, 0.94],
])
f_table = np.array([
    [5.0, 4.6, 4.2, 4.0, 4.0],
    [6.7, 5.8, 5.0, 4.8, 4.7],
    [8.1, 6.8, 5.3, 5.1, 5.3],
    [9.1, 7.0, 5.0, 4.9, 5.2],
    [9.8, 7.1, 4.9, 4.8, 5.0],
]) * 1e22 # SI


if __name__ == "__main__":
    with open(f"{prefix}/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    dexrt_config_path = f"{prefix}/" + config["dexrt_config_path"]

    with open(dexrt_config_path, "r") as f:
        dexrt_config = yaml.load(f, Loader=yaml.Loader)
    atmos = netCDF4.Dataset(f"{prefix}/" + dexrt_config["atmos_path"], "r")

    ion_interp = RegularGridInterpolator(
        (temperature_grid, pressure_grid),
        ionisation_table,
        bounds_error=False,
    )
    f_interp = RegularGridInterpolator(
        (temperature_grid, pressure_grid),
        f_table,
        bounds_error=False,
    )

    voxel_scale = np.float64(atmos["voxel_scale"][...])

    pg = np.array(atmos["pressure"][...])
    temperature = np.array(atmos["temperature"][...])
    ionisation_fraction = ion_interp(np.stack([temperature, pg], axis=2))
    f_value = f_interp(np.stack([temperature, pg], axis=2))
    ne = pg / (const.k_B.value * temperature) / (1 + 1.1 / ionisation_fraction)
    n2 = ne**2 / f_value
    n2_cm = n2 * 1e-6

    delta_lambda_max = 0.2
    lambda0 = 656.4691622298104
    delta_lambda = np.linspace(-delta_lambda_max, delta_lambda_max, 251)
    delta_nu = const.c.value / (lambda0 * 1e-9)**2 * (delta_lambda*1e-9)
    delta_nuD = 1.0 / (lambda0 * 1e-9) * (
        2.0 * const.k_B.value * temperature / const.m_p.value + np.array(atmos["vturb"][...])**2
    )**0.5
    # phi = 1.0 / (np.sqrt(np.pi) * delta_nuD[:, :, None]) * np.exp(-(delta_nu[None, None, :] / delta_nuD[:, :, None])**2)
    # NOTE(cmo): We go from +x to -x
    prom_dop_shift = -np.array(atmos["vx"][...]) / (lambda0 * 1e-9)
    # We go from -z to +z
    fil_dop_shift = np.array(atmos["vz"][...]) / (lambda0 * 1e-9)

    # TODO(cmo): This is fixed at 10 Mm
    # W = 0.5 * (1.0 - np.sqrt(1.0 - const.R_sun.value**2 / (const.R_sun.value**2 + 10e6**2)))
    # source_fn = W * 0.17 * 4.077e-5

    source_fn_cgs = 3.22e-6 # erg/s/cm2/sr/Hz
    # source_fn = (source_fn_cgs << u.Unit("erg/(s cm2 sr Hz)")).to("J / (s m2 Hz sr)", equivalencies=u.spectral_density(wav=(lambda0 << u.nm))).value
    source_fn = (source_fn_cgs << u.Unit("erg/(s cm2 sr Hz)")).to("kW / (m2 nm sr)", equivalencies=u.spectral_density(wav=(lambda0 << u.nm))).value
    # kappa_cgs = 1.7e-2 * n2_cm[:, :, None] * phi_prom
    # kappa = kappa_cgs * 100

    @njit
    def integrate_ray(source_fn, kappa, voxel_scale, background=0.0):
        intens = np.ones(kappa.shape[1]) * background
        tau_tot = np.zeros(kappa.shape[1])
        for k in range(kappa.shape[0], -1, -1):
            if np.isnan(kappa[k, 0]):
                continue
            dtau = kappa[k] * voxel_scale
            tau_tot += dtau
            edt = np.exp(-dtau)
            intens *= edt
            intens += source_fn * (1.0 - edt)
        return intens, tau_tot

    # prom proj
    intens_prom = np.zeros((temperature.shape[0], delta_nu.shape[0]))
    tau_prom = np.zeros((temperature.shape[0], delta_nu.shape[0]))
    for la in tqdm(range(delta_nu.shape[0])):
        phi_plane = 1.0 / (np.sqrt(np.pi) * delta_nuD[:, :, None]) * np.exp(-((delta_nu[None, None, la] + prom_dop_shift[:, :, None]) / delta_nuD[:, :, None])**2)
        kappa_cgs = 1.7e-2 * n2_cm[:, :, None] * phi_plane
        kappa = kappa_cgs * 100
        for row in range(temperature.shape[0]):
            i_sample, tau_sample = integrate_ray(source_fn, kappa[row, :, :], voxel_scale, background=0.0)
            intens_prom[row, la] = i_sample[0]
            tau_prom[row, la] = tau_sample[0]

    bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"])
    background_intens = bc_ctx.compute_rays(wavelengths=(lambda0+delta_lambda), mus=1.0)
    background_intens = (background_intens << u.Unit("J / (s m2 Hz sr)")).to(u.Unit("kW / (m2 nm sr)"), equivalencies=u.spectral_density(wav=(lambda0+delta_lambda) << u.nm)).value

    # fil proj
    intens_fil = np.zeros((temperature.shape[1], delta_nu.shape[0]))
    tau_fil = np.zeros((temperature.shape[1], delta_nu.shape[0]))
    for la in tqdm(range(delta_nu.shape[0])):
        phi_plane = 1.0 / (np.sqrt(np.pi) * delta_nuD[:, :, None]) * np.exp(-((delta_nu[None, None, la] + fil_dop_shift[:, :, None]) / delta_nuD[:, :, None])**2)
        kappa_cgs = 1.7e-2 * n2_cm[:, :, None] * phi_plane
        kappa = kappa_cgs * 100
        for col in range(temperature.shape[1]):
            i_sample, tau_sample = integrate_ray(
                source_fn,
                np.ascontiguousarray(kappa[::-1, col, :]),
                voxel_scale,
                background=background_intens[la],
            )
            intens_fil[col, la] = i_sample[0]
            tau_fil[col, la] = tau_sample[0]

    dataset = xr.Dataset(
        {
            "intens_prom": (("z", "wavelength"), intens_prom),
            "tau_prom": (("z", "wavelength"), tau_prom),
            "intens_fil": (("x", "wavelength"), intens_fil),
            "tau_fil": (("x", "wavelength"), tau_fil),
        },
        coords={
            "wavelength": lambda0 + delta_lambda
        }
    )
    dataset.to_netcdf(f"{file_id}_hea2015_proxy.nc")



