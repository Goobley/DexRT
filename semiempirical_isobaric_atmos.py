import netCDF4 as ncdf
import lightweaver as lw
from lightweaver.rh_atoms import H_atom, H_6_atom
import promweaver as pw
import numpy as np
from dexrt.config_schemas.dexrt import DexrtNgConfig, DexrtNonLteConfig, AtomicModelConfig, DexrtLteConfig, DexrtSystemConfig, DexrtOutputConfig, DexrtMipConfig
from dexrt.write_config import write_config

pressure_val = 1e-2
# lyman_cont = "pw"
lyman_cont = "obs"
# lyman_cont = "bb"
# lyman_cont = "full_bb"
shape = "square"
shape = "circle"
prefix = f"rad_loss_p{pressure_val:.0e}_{lyman_cont}_{shape}_falling"
if __name__ == '__main__':
    nc = ncdf.Dataset(f"RadLossTest/{prefix}_atmos.nc", "w", format="NETCDF4")

    atomic_models = pw.default_atomic_models()
    atomic_models[0] = H_6_atom()
    bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"], atomic_models=atomic_models, prd=True, Nthreads=6)
    boundary_wavelengths = bc_ctx.spect.wavelength
    boundary_wavelengths = np.unique(np.sort(np.concatenate([bc_ctx.spect.wavelength, np.arange(20.0, 4000.0, 2.0)])))
    tabulated = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20), wavelength=boundary_wavelengths)
    I_with_zero = np.zeros((tabulated["I"].shape[0], tabulated["I"].shape[1] + 1))
    I_with_zero[:, 1:] = tabulated["I"][...]
    tabulated["I"] = I_with_zero
    tabulated["mu_grid"] = np.concatenate([[0], tabulated["mu_grid"]])

    if lyman_cont == "bb":
        mask = tabulated["wavelength"] < 91.2
        tabulated["I"][mask, 1:] = lw.planck(8000, tabulated["wavelength"][mask])[:, None]
    elif lyman_cont == "full_bb":
        tabulated["I"][:, 1:] = lw.planck(6500, tabulated["wavelength"])[:, None]
    elif lyman_cont == "obs":
        mask = boundary_wavelengths < 91.2
        waves_to_compute = boundary_wavelengths[mask]
        lyman_rad = np.zeros_like(waves_to_compute)
        tembri = np.genfromtxt("RadLossTest/tembri.dat")
        tembri_waves = tembri[:, 0] * 1e3
        brightness_temps = np.ascontiguousarray(tembri[:, 1])
        for i, w in enumerate(waves_to_compute):
            lyman_rad[i] = lw.planck(np.interp(w, tembri_waves, brightness_temps), w)
        tabulated["I"][mask, 1:]  = lyman_rad[:, None]

    mu_dim = nc.createDimension("prom_bc_mu", tabulated["mu_grid"].shape[0])
    bc_wl_dim = nc.createDimension("prom_bc_wavelength", tabulated["wavelength"].shape[0])
    mu_min = nc.createVariable("prom_bc_mu_min", "f4")
    mu_max = nc.createVariable("prom_bc_mu_max", "f4")
    mu_min[...] = tabulated["mu_grid"][0]
    mu_max[...] = tabulated["mu_grid"][-1]
    bc_wavelength = nc.createVariable("prom_bc_wavelength", "f4", ("prom_bc_wavelength",))
    bc_wavelength[...] = tabulated["wavelength"]

    bc_I = nc.createVariable("prom_bc_I", "f4", ("prom_bc_wavelength", "prom_bc_mu"))
    for la in range(tabulated["wavelength"].shape[0]):
        bc_I[la, :] = lw.convert_specific_intensity(
            tabulated["wavelength"][la],
            tabulated["I"][la, :],
            outUnits="kW / (m2 nm sr)"
        ).value

    atmos_size = 256 + 64
    x_dim = nc.createDimension("x", atmos_size)
    z_dim = nc.createDimension("z", atmos_size)
    index_order = ("z", "x")
    temperature = nc.createVariable("temperature", "f4", index_order)
    ne = nc.createVariable("ne", "f4", index_order)
    nh_tot = nc.createVariable("nh_tot", "f4", index_order)
    vturb = nc.createVariable("vturb", "f4", index_order)
    pressure = nc.createVariable("pressure", "f4", index_order)
    vx = nc.createVariable("vx", "f4", index_order)
    vy = nc.createVariable("vy", "f4", index_order)
    vz = nc.createVariable("vz", "f4", index_order)
    scale = nc.createVariable("voxel_scale", "f4")
    altitude = nc.createVariable("offset_z", "f4")
    offset_x = nc.createVariable("offset_x", "f4")
    offset_y = nc.createVariable("offset_y", "f4")

    # NOTE(cmo): We should have a margin around the outside of the model to avoid issues with the CLAMP modes

    # prom_cell_dim = 246
    prom_cell_dim = 256
    atmos_size_m = 1.0e6
    # atmos_size_m = 2e5
    scale[...] = atmos_size_m / prom_cell_dim
    altitude[...] = 10.0e6
    offset_x[...] = -0.5 * atmos_size_m
    offset_y[...] = 0.0
    high_temp_val = 300e3
    temp_val = 12e3
    temperature[...] = high_temp_val
    coronal_pres_val = pressure_val * 0.1
    pressure[...] = coronal_pres_val

    if shape == "circle":
        centre = np.array([0.5 * atmos_size, 0.5 * atmos_size])
        mgrid = np.mgrid[:atmos_size, :atmos_size]
        prom_mask = np.sum((mgrid - centre[:, None, None])**2, axis=0) < (0.5*prom_cell_dim)**2
        temp = temperature[...]
        temp[prom_mask] = temp_val
        temperature[...] = temp
        temp = pressure[...]
        temp[prom_mask] = pressure_val
        pressure[...] = temp
    elif shape == "square":
        prom_slice = slice(atmos_size // 2 - prom_cell_dim // 2, atmos_size // 2 + prom_cell_dim // 2)
        temperature[prom_slice, prom_slice] = temp_val
        pressure[prom_slice, prom_slice] = pressure_val

    # NOTE(cmo): Approximate ionisation fraction
    X = 0.9
    nh_val = pressure[...] / (lw.KBoltzmann * temperature[...] * (1.0 + X))
    nh_tot[...] = nh_val
    ne[...] = X * nh_val
    vturb[...] = 5e3

    vx[...] = 0.0
    vy[...] = 0.0
    vz[...] = -60e3

    nc.close()

    config_sparse = DexrtNonLteConfig(
        atmos_path=f"{prefix}_atmos.nc",
        output_path=f"{prefix}.nc",
        sparse_calculation=True,
        threshold_temperature=250e3,
        atoms={
            # "Ca": AtomicModelConfig(
            #     path="CaII.yaml",
            #     # initial_populations="ZeroRadiation",
            # ),
            "H": AtomicModelConfig(
                path="H_6.yaml",
                # initial_populations="ZeroRadiation",
            )
        },
        boundary_type="Promweaver",
        conserve_charge=True,
        conserve_pressure=True,
        max_iter=900,
        max_cascade=4,
        pop_tol=9e-4,
        store_J_on_cpu=True,
        system=DexrtSystemConfig(
            mem_pool_gb=3,
        ),
        output=DexrtOutputConfig(
            sparse=True
        ),
        mip_config=DexrtMipConfig(
            opacity_threshold=0.1
        ),
        final_dense_fs=False,
        rad_loss="Integrated",
        snapshot_frequency=10,
        ng_config=DexrtNgConfig(
            enable=False,
            threshold=6e-3,
            lower_threshold=1e-3,
        ),
    )
    write_config(config_sparse, f"RadLossTest/{prefix}.yaml")
    print(f"Wrote config to {prefix}.yaml")