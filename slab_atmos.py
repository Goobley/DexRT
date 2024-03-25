import netCDF4 as ncdf
import lightweaver as lw

if __name__ == '__main__':
    atmos = ncdf.Dataset("build/atmos.nc", "w", format="NETCDF4")

    atmos_size = 512
    x_dim = atmos.createDimension("x", atmos_size)
    z_dim = atmos.createDimension("z", atmos_size)
    temperature = atmos.createVariable("temperature", "f4", ("x", "z"))
    ne = atmos.createVariable("ne", "f4", ("x", "z"))
    nh_tot = atmos.createVariable("nh_tot", "f4", ("x", "z"))
    vturb = atmos.createVariable("vturb", "f4", ("x", "z"))
    pressure = atmos.createVariable("pressure", "f4", ("x", "z"))
    scale = atmos.createVariable("voxel_scale", "f4")

    scale[...] = 30e6 / atmos_size
    temp_val = 5000.0
    temperature[...] = temp_val
    pres_val = 0.1
    pressure[...] = pres_val
    # NOTE(cmo): Approximate ionisation fraction
    X = 0.1
    nh_val = pres_val / (lw.KBoltzmann * temp_val * (1.0 + X))
    nh_tot[...] = nh_val
    ne[...] = X * nh_val
    vturb[...] = 5e3

    atmos.close()