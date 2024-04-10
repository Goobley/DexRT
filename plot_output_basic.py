import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from dataclasses import dataclass
from tqdm import tqdm

PLOT_COMPARISON = False

if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    im = np.array(ds["image"][...])

    J = np.swapaxes(im, 0, -1)

    plt.ion()
    fig = plt.figure(layout="constrained", figsize=(10, 6))
    plt.imshow(J)
    plt.title("No mips")


