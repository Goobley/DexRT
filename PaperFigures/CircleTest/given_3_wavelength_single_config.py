import lightweaver as lw
from matplotlib import pyplot as plt
import numpy as np
import netCDF4

from dexrt_py.dexrt.config_schemas.dexrt import DexrtGivenFsConfig
from dexrt_py.dexrt.write_config import write_config

SIZE = 1024
NUM_WAVE = 3

def draw_disk(arr, centre, radius, color):
    X, Y = np.mgrid[:arr.shape[0], :arr.shape[1]]
    dist_x = X - centre[0]
    dist_y = Y - centre[1]

    mask = ((dist_x**2 + dist_y**2) <= radius * radius)
    arr[mask, :] = color[None, :]

def compute_model():
    eta = np.zeros((SIZE, SIZE, NUM_WAVE), dtype=np.float32)
    chi = np.zeros((SIZE, SIZE, NUM_WAVE), dtype=np.float32)

    centre = (int(SIZE // 2), int(SIZE // 2))
    c = np.array([centre[0], centre[1]])
    factor = 1e2
    # scale1 = 100
    # scale2 = 4
    # scale3 = 5
    scale3 = 20
    color = np.array([8, 8, 8]) * factor
    draw_disk(eta, c, scale3, color)

    color_chi = np.array([1, 1, 1]) * factor
    draw_disk(chi, c, scale3, color_chi)

    # NOTE(cmo): Blocker
    # color_chi = np.array([1, 1, 1]) * factor
    # chi[centre[0]+scale1:centre[0]+scale1+scale2, centre[1]+scale1:centre[1]+2*scale1, :] = color_chi[None, None, :]
    # eta[centre[0]+scale1:centre[0]+scale1+scale2, centre[1]+scale1:centre[1]+2*scale1, :] = color_chi[None, None, :]

    # chi = np.ascontiguousarray(np.swapaxes(chi, 0, 1))
    # eta = np.ascontiguousarray(np.swapaxes(eta, 0, 1))
    chi = np.ascontiguousarray(chi)
    eta = np.ascontiguousarray(eta)
    return chi, eta

atmos = netCDF4.Dataset("build/circle_test.nc", "w", format="NETCDF4")

chi, eta = compute_model()

x_dim = atmos.createDimension("x", SIZE)
z_dim = atmos.createDimension("z", SIZE)
wave_dim = atmos.createDimension("wavelength", NUM_WAVE)
index_order = ("z", "x", "wavelength")
eta_model = atmos.createVariable("eta", "f4", index_order)
eta_model[...] = eta
chi_model = atmos.createVariable("chi", "f4", index_order)
chi_model[...] = chi
scale = atmos.createVariable("voxel_scale", "f4")
scale[...] = 1.0

atmos.close()

config = DexrtGivenFsConfig(
    atmos_path="circle_test.nc",
    output_path="circle_test_out.nc",
)
write_config(config, "build/circle_test.yaml")