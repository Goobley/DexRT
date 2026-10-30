"""Audit the effect of the global edge bins on each continuum for a set of
atoms: builds DexRT's merged wavelength grid (via a quick dexrt_sc LTE run)
and compares the edge weight and recombination/photoionisation integrals
against continuum-local trapezoidal weights.
Usage: python cont_edge_bin_audit.py DEXRT_SC WORKDIR ATMOS ATOM.yaml [ATOM.yaml ...]"""
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml

dexrt_sc, work, atmos = [Path(p).resolve() for p in sys.argv[1:4]]
atoms = [Path(p).resolve() for p in sys.argv[4:]]
work.mkdir(parents=True, exist_ok=True)
tag = "_".join(a.stem for a in atoms)
conf = {
    "mode": "Lte", "atmos_path": str(atmos), "output_path": str(work / f"audit_{tag}.nc"),
    "atoms": {f"A{i}": {"path": str(a), "treatment": "Active"} for i, a in enumerate(atoms)},
    "boundary_type": "Zero", "store_J_on_cpu": False, "system": {"mem_pool_gb": 2.0},
}
(work / f"audit_{tag}.yaml").write_text(yaml.dump(conf))
with open(work / f"audit_{tag}.log", "w") as log:
    subprocess.run([str(dexrt_sc), "--config", str(work / f"audit_{tag}.yaml")], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
wl = np.asarray(ncdf.Dataset(work / f"audit_{tag}.nc")["wavelength"][...]).astype(np.float64)
gbin = np.empty_like(wl)
gbin[1:-1] = 0.5 * (wl[2:] - wl[:-2])
gbin[0] = 0.5 * (wl[1] - wl[0])
gbin[-1] = 0.5 * (wl[-1] - wl[-2])

h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
print(f"== {' + '.join(a.name for a in atoms)}: {len(wl)} wavelengths")
for a in atoms:
    model = yaml.safe_load(a.read_text())
    for cont in model["continua"]:
        tab = np.array(cont["value"])
        lo, hi = tab[0, 0], tab[-1, 0]
        m = (wl >= lo * (1 - 1e-6)) & (wl <= hi * (1 + 1e-6))
        lam = wl[m]
        w_dex = gbin[m]
        w_loc = w_dex.copy()
        w_loc[0] = 0.5 * (lam[1] - lam[0])
        w_loc[-1] = 0.5 * (lam[-1] - lam[-2])
        sig = np.interp(lam, tab[:, 0], tab[:, 1])
        out = []
        for T in [6000.0, 10000.0]:
            # spontaneous recombination ~ int sigma 2c/lam^4 exp(-hc/lam kT) dlam
            f = sig * lam ** -4 * np.exp(-h * c / (lam * 1e-9 * k * T))
            out.append(np.sum(f * w_dex) / np.sum(f * w_loc))
        # photoionisation by a flat J (in lambda): int sigma lam dlam
        g = sig * lam
        pi = np.sum(g * w_dex) / np.sum(g * w_loc)
        print(f"  {model['element']['symbol']} {cont['transition']}: edge {hi:8.2f} nm, edge bin {w_dex[-1]:8.3f} vs {w_loc[-1]:7.3f} nm"
              f" | recomb x{out[0]:.3f} (6kK) x{out[1]:.3f} (10kK) | photoion (flat J) x{pi:.3f}")
