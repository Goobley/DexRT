import netCDF4 as ncdf
import lightweaver as lw
import promweaver as pw
import numpy as np

if __name__ == '__main__':
    atmos = ncdf.Dataset("build/atmos.nc", "w", format="NETCDF4")

    # bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"])
    bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["Ca"])
    tabulated = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    I_with_zero = np.zeros((tabulated["I"].shape[0], tabulated["I"].shape[1] + 1))
    I_with_zero[:, 1:] = tabulated["I"][...]
    tabulated["I"] = I_with_zero
    tabulated["mu_grid"] = np.concatenate([[0], tabulated["mu_grid"]])

    mu_dim = atmos.createDimension("prom_bc_mu", tabulated["mu_grid"].shape[0])
    bc_wl_dim = atmos.createDimension("prom_bc_wavelength", tabulated["wavelength"].shape[0])
    mu_min = atmos.createVariable("prom_bc_mu_min", "f4")
    mu_max = atmos.createVariable("prom_bc_mu_max", "f4")
    mu_min[...] = tabulated["mu_grid"][0]
    mu_max[...] = tabulated["mu_grid"][-1]
    bc_wavelength = atmos.createVariable("prom_bc_wavelength", "f4", ("prom_bc_wavelength",))
    bc_wavelength[...] = tabulated["wavelength"]
    bc_I = atmos.createVariable("prom_bc_I", "f4", ("prom_bc_wavelength", "prom_bc_mu"))

    for la in range(tabulated["wavelength"].shape[0]):
        bc_I[la, :] = lw.convert_specific_intensity(
            tabulated["wavelength"][la],
            tabulated["I"][la, :],
            outUnits="kW / (m2 nm sr)"
        ).value


    atmos_size = 128
    atmos_size_x = 8192
    x_dim = atmos.createDimension("x", atmos_size_x)
    z_dim = atmos.createDimension("z", atmos_size)
    index_order = ("z", "x")
    temperature = atmos.createVariable("temperature", "f4", index_order)
    ne = atmos.createVariable("ne", "f4", index_order)
    nh_tot = atmos.createVariable("nh_tot", "f4", index_order)
    vturb = atmos.createVariable("vturb", "f4", index_order)
    pressure = atmos.createVariable("pressure", "f4", index_order)
    vx = atmos.createVariable("vx", "f4", index_order)
    vy = atmos.createVariable("vy", "f4", index_order)
    vz = atmos.createVariable("vz", "f4", index_order)
    scale = atmos.createVariable("voxel_scale", "f4")
    altitude = atmos.createVariable("offset_z", "f4")
    offset_x = atmos.createVariable("offset_x", "f4")

    atmos_size_m = 10.0e6
    scale[...] = atmos_size_m / atmos_size
    altitude[...] = 10.0e6
    offset_x[...] = -0.5 * atmos_size_m * atmos_size_x / atmos_size
    temp_val = 8000
    temperature[...] = temp_val
    pres_val = 0.1
    pressure[...] = pres_val
    # NOTE(cmo): Approximate ionisation fraction
    X = 0.1
    nh_val = pres_val / (lw.KBoltzmann * temp_val * (1.0 + X))
    nh_tot[...] = nh_val
    ne[...] = X * nh_val
    vturb[...] = 5e3

    vx[...] = 0.0
    vy[...] = 0.0
    vz[...] = 0.0

    atmos.close()