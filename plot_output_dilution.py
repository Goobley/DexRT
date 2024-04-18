import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

def classical_dilution(h):
    Rs = const.R_sun.value
    return 0.5 * (1.0 - np.sqrt(1.0 - Rs**2 / (Rs + h)**2))


if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    dex_J = np.array(ds["image"][...])
    dex_wave = np.array(ds["wavelength"][...])
    dex_eta = np.array(ds["eta"][...])
    dex_chi = np.array(ds["chi"][...])
    dex_pops = np.array(ds["pops"][...])

    ds = netCDF4.Dataset("build/atmos.nc", "r")
    nz = ds.dimensions["z"].size
    z_grid = np.ascontiguousarray(((np.arange(nz, dtype=np.float64) + 0.5) * ds["voxel_scale"][...]))
    sample_idx = ds["ne"].shape[1] // 2
    temperature = np.ascontiguousarray(ds["temperature"][:, sample_idx][::-1]).astype(np.float64)
    ne = np.ascontiguousarray(ds["ne"][:, sample_idx][::-1]).astype(np.float64)
    nhtot = np.ascontiguousarray(ds["nh_tot"][:, sample_idx][::-1]).astype(np.float64)
    vturb = np.ascontiguousarray(ds["vturb"][:, sample_idx][::-1]).astype(np.float64)
    altitude = float(ds["offset_z"][...])
    z_grid += altitude

    dilutions = classical_dilution(z_grid)

    plt.ion()
    plt.figure()
    plt.plot(z_grid, dilutions, label="Expected")
    sample_idx = 5
    plt.plot(z_grid, dex_J[29, :, sample_idx], label="Static solver")
    plt.plot(z_grid, dex_J[200, :, sample_idx], label="Dynamic solver")
    plt.legend()


