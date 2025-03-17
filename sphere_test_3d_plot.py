import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()

def solid_angle_sphere(radius, distance):
    # https://math.stackexchange.com/questions/73238/calculating-solid-angle-for-a-sphere-in-space
    return np.where(
        distance < radius,
        4.0 * np.pi,
        2.0 * np.pi * (1.0 - np.sqrt(distance**2 - radius**2) / distance)
        # 2.0 * np.pi * (1.0 - np.cos(np.arcsin(radius / distance)))
    )

if __name__ == "__main__":
    # ds = xr.open_dataset("build/sphere_test_out_2x_oct_12.nc")
    # ds = xr.open_dataset("build/sphere_test_out_3x_oct_6.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_oct_6_tri.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_trapez_6_tri.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_hp1_12.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_hp1_12_tri.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_hp_6_tri.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_hp_12_tri.nc")
    # ds = xr.open_dataset("build/sphere_test_out_2x_hp1_18.nc")
    # ds = xr.open_dataset("build/sphere_test_out_A2.nc")
    # ds = xr.open_dataset("build/sphere_test_out_A4.nc")
    ds = xr.open_dataset("build/sphere_test_out_A8.nc")

    size = ds.J.shape[1]

    z_cut = size // 2
    z_cut = 80
    J_slice = ds.J[2, :, z_cut, :].values
    XX, YY = np.mgrid[:size, :size].astype(np.float64)
    XX += 0.5
    YY += 0.5
    centre = (size // 2, size // 2,  size // 2)
    dists = np.sqrt((XX - centre[0])**2 + (YY - centre[1])**2 + (z_cut + 0.5 - centre[2])**2)

    source_fn = 8.0
    solid_angle = solid_angle_sphere(18.0, dists)
    theoretical_intensity = solid_angle * source_fn / (4.0 * np.pi)

    fig, ax = plt.subplots(1, 2, layout='constrained')
    ax[0].imshow(J_slice**(1.0 / 3.0))
    ax[0].set_title("Cut")
    mappable = ax[1].imshow(J_slice / theoretical_intensity)
    ax[1].set_title("RC/Theory")
    fig.colorbar(mappable)

    line_cut = size // 2 + 15
    fig, ax = plt.subplots(2, 1)
    ax[0].semilogy(J_slice[line_cut])
    ax[0].semilogy(theoretical_intensity[line_cut])

    ax[1].plot((J_slice[line_cut] - theoretical_intensity[line_cut]) / theoretical_intensity[line_cut])
    ax[1].set_yscale('symlog', linthresh=1e-2)
    ax[1].set_ylim(0, 1)




