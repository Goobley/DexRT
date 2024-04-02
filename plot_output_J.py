import numpy as np
import netCDF4
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm


if __name__ == "__main__":
    ds = netCDF4.Dataset("build/output.nc")
    J = np.array(ds["image"][...])

    plt.ion()
    plt.figure()
    plt.imshow(J[100])
