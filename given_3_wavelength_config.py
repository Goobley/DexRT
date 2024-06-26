import lightweaver as lw
from matplotlib import pyplot as plt
import numpy as np
import netCDF4

from dexrt_py.dexrt.config_schemas.dexrt import DexrtGivenFsConfig
from dexrt_py.dexrt.write_config import write_config

SIZE = 1024
NUM_WAVE = 3

def draw_disk(arr, centre, radius, color):
    X, Y = np.mgrid[:SIZE, :SIZE]
    dist_x = X - centre[0]
    dist_y = Y - centre[1]

    mask = ((dist_x**2 + dist_y**2) <= radius * radius)
    arr[mask, :] = color[None, :]

def compute_model():
    eta = np.zeros((SIZE, SIZE, NUM_WAVE), dtype=np.float32)
    chi = np.zeros((SIZE, SIZE, NUM_WAVE), dtype=np.float32)

    centre = int(SIZE // 2)
    c = np.array([centre+400, centre])
    color = np.array([0.0, 0.0, 2.0])
    draw_disk(eta, c, 40, color)

    c = np.array([centre-400, centre])
    color = np.array([2.0, 0.0, 0.0])
    draw_disk(eta, c, 40, color)

    c = np.array([centre, centre-400])
    color = np.array([0.0, 0.0, 0.5])
    draw_disk(eta, c, 40, color)

    c = np.array([centre, centre+400])
    color = np.array([0.5, 0.0, 0.0])
    draw_disk(eta, c, 40, color)

    c = np.array([200, 200])
    color = np.array([0.0, 3.0, 0.0])
    draw_disk(eta, c, 40, color)

    c = np.array([SIZE-200, 200])
    color = np.array([0.0, 0.0, 3.0])
    draw_disk(eta, c, 40, color)

    c = np.array([400, 700])
    color = np.array([3.0, 0.0, 3.0])
    draw_disk(eta, c, 40, color)

    chi[...] = 1e-10
    bg = 1e-10
    c = np.array([centre+400, centre])
    color = np.array([bg, bg, 0.5])
    draw_disk(chi, c, 40, color)

    c = np.array([centre-400, centre])
    color = np.array([0.5, bg, bg])
    draw_disk(chi, c, 40, color)

    c = np.array([centre, centre-400])
    color = np.array([bg, bg, 0.5])
    draw_disk(chi, c, 40, color)

    c = np.array([centre, centre+400])
    color = np.array([0.5, bg, bg])
    draw_disk(chi, c, 40, color)

    c = np.array([centre+340, centre])
    color = np.array([0.2, 0.2, 0.2])
    draw_disk(chi, c, 6, color)

    c = np.array([centre-340, centre])
    color = np.array([0.2, 0.2, 0.2])
    draw_disk(chi, c, 6, color)

    box_size = 250
    chi_r = 1e-4
    chi[centre-box_size:centre+box_size, centre-box_size:centre+box_size, 0] = chi_r
    chi[centre-box_size:centre+box_size, centre-box_size:centre+box_size, 1:] = 1e2 * chi_r

    c = np.array([200, 200])
    color = np.array([bg, 1.0, bg])
    draw_disk(chi, c, 40, color)

    c = np.array([SIZE-200, 200])
    color = np.array([bg, bg, 1.0])
    draw_disk(chi, c, 40, color)

    c = np.array([400, 700])
    color = np.array([1.0, bg, 1.0])
    draw_disk(chi, c, 40, color)
    chi = np.ascontiguousarray(np.swapaxes(chi, 0, 1))
    eta = np.ascontiguousarray(np.swapaxes(eta, 0, 1))
    return chi, eta

atmos = netCDF4.Dataset("build/disco_test.nc", "w", format="NETCDF4")

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
    atmos_path="disco_test.nc",
    output_path="disco_test_out.nc",
)
write_config(config, "build/disco_test.yaml")