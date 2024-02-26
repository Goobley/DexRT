import numpy as np
import netCDF4
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    im = np.array(ds["image"][...])

    J = np.mean(im, axis=2)
    canvas = np.copy(J)

    plt.ion()
    plt.imshow(canvas, origin="lower", interpolation="nearest")

