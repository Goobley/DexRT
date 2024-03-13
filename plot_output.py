import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from dataclasses import dataclass
from tqdm import tqdm

PLOT_COMPARISON = False

@dataclass
class Ray:
    o: np.ndarray
    d: np.ndarray

def ray_sphere_path_length(ray, centre, radius):
    # NOTE(cmo): RT1W implementation
    oc = ray.o - centre
    a = ray.d @ ray.d
    b = 2.0 * oc @ ray.d
    c = oc @ oc - radius**2
    discriminant = b**2 - 4.0 * a * c

    if discriminant <= 0.0:
        return 0.0

    sqrt_d = np.sqrt(discriminant)
    t0_numerator = (-b - sqrt_d)
    t1_numerator = (-b + sqrt_d)

    if t0_numerator < 0.0 and t1_numerator < 0.0:
        return 0.0
    elif t0_numerator * t1_numerator < 0.0:
        # NOTE(cmo): inside, so avoid double counting
        return 0.5 * sqrt_d / a

    return sqrt_d / a

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    im = np.array(ds["image"][...])

    J = np.mean(im, axis=2)
    canvas = np.copy(J)


    if PLOT_COMPARISON:
        plt.ion()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.tanh(canvas), origin="lower", interpolation="nearest")
        ax[0].set_title("Radiance Cascades")
        lw_J = np.load("../LwDexVerification/J_10th.npy")
        ax[1].imshow(np.tanh(lw_J), origin="lower", interpolation="nearest")
        ax[1].set_title("Linear Short Characteristics 10ray/octant (13th order)")
    else:
        plt.ion()
        fig = plt.figure(layout="constrained", figsize=(10, 6))
        figname = "NoFix_NoMips"
        ax = fig.subplot_mosaic(
            """
            AB
            AC
            """)
        ax["A"].imshow(np.tanh(canvas), origin="lower", interpolation="nearest")
        # ax["A"].set_title("Bilinear Fix + Branching (No mipmaps)")
        ax["A"].set_title("\"Classical\" RCs (No mipmaps)")

        centre = 512
        start = [centre, 0]
        end = [centre, 1023]
        # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
        line1 = profile_line(canvas, start, end)
        start = [0, centre]
        end = [1023, centre]
        # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
        line2 = profile_line(canvas, start, end)

        start = [centre - int(centre/np.sqrt(2)), centre - int(centre/np.sqrt(2))]
        end = [centre + int(centre / np.sqrt(2)), centre + int(centre/np.sqrt(2)) - 1.414]
        # ax["A"].plot([start[0], end[0]], [start[1], end[1]])
        line3 = profile_line(canvas, start, end)

        ax["B"].plot(line1[:, 0])
        ax["B"].plot(line2[:, 0])
        ax["B"].plot(line3[:, 0])
        ax["B"].set_title("Intensity Slices")

        coord = np.arange(1024, dtype=np.float64)
        r_coord = np.abs(coord - centre)
        theory_solid = np.zeros_like(coord)

        PEAK = 1.0
        tau = 1.0 / 80.0 * 40.0
        # tau = 1.0 * 40.0
        source_fn = 10.0
        source_factor = -np.expm1(-tau)
        PEAK = source_factor * source_fn
        RADIUS = 40
        theory_solid[:] = PEAK
        theory_solid[r_coord >= RADIUS] = 2.0 * np.arcsin(RADIUS / r_coord[r_coord >= RADIUS]) * PEAK / (2.0 * np.pi)


        simple_model = np.zeros(1024)

        centre = np.array([512, 512], dtype=np.float64)
        radius = 40.0

        num_rays = 1024
        thetas = (np.arange(num_rays) + 0.5) * 2.0 * np.pi / num_rays

        dirs = np.hstack((np.cos(thetas)[:, None], np.sin(thetas)[:, None]))
        chi = 1.0 / 80.0
        x_coords = np.arange(1024) + 0.0
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        for x_idx, x_coord in tqdm(enumerate(x_coords), total=x_coords.shape[0]):
            for d_idx in range(dirs.shape[0]):
                coord = (x_coord - 512) * inv_sqrt2 + 512
                ray = Ray(o=np.array([coord, coord]), d=dirs[d_idx, :])
                path_length = ray_sphere_path_length(ray, centre, radius)
                if path_length <= 0.0:
                    continue
                tau = chi * path_length
                source_factor = -np.expm1(-tau)
                simple_model[x_idx] += source_factor * source_fn
        simple_model /= num_rays

        theory = simple_model
        coord = x_coords

        ax["B"].plot(coord, theory, 'k--', label="Theory Diffuse")
        ax["B"].plot(coord, theory_solid, 'k-.', label="Theory Solid")
        ax["B"].set_yscale('log')
        ax["B"].legend()

        error1 = (line1[:, 0] - theory) / theory
        error2 = (line2[:, 0] - theory) / theory
        error3 = (line3[:, 0] - theory) / theory
        ax["C"].plot(coord, error1)
        ax["C"].plot(coord, error2)
        ax["C"].plot(coord, error3)
        ax["C"].set_yscale("symlog", linthresh=1e-2)
        ax["C"].set_title("Relative Error")

        fig.savefig(f"{figname}.png", dpi=300)


