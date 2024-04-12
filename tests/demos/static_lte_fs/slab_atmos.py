import netCDF4 as ncdf
import lightweaver as lw

if __name__ == '__main__':
    atmos = ncdf.Dataset("../../build/atmos_static_lte_fs.nc", "w", format="NETCDF4")

    atmos_size = 512
    x_dim = atmos.createDimension("x", atmos_size)
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

    vx[...] = 0.0
    vy[...] = 0.0
    vz[...] = 0.0

    atmos.close()