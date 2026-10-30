"""Formal solution (+ Gamma) throughput of dexrt_sc vs Lightweaver on the same
2D LTE-populations H_6 problem (periodic x, zero z boundaries, mirrored Stepan
quadrature, piecewise linear SC with linear interpolation).
Usage: python bench_vs_lw.py WORKDIR N LW_THREADS DEXRT_SC [DEXRT_SC_CPU OMP_THREADS]"""
import re
import subprocess
import sys
import time
import os
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom

work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2])
lw_threads = int(sys.argv[3])
dexrt_gpu = Path(sys.argv[4]).resolve()
dexrt_cpu = Path(sys.argv[5]).resolve() if len(sys.argv) > 5 else None
omp_threads = int(sys.argv[6]) if len(sys.argv) > 6 else lw_threads
repo = Path(__file__).resolve().parents[2]

nx = nz = n
dx = 100e3 * 64 / n
KB = 1.380649e-23
zz, xx = np.meshgrid((np.arange(nz) + 0.5) / nz, (np.arange(nx) + 0.5) / nx, indexing="ij")
T = 6000.0 + 4000.0 * np.exp(-((xx - 0.4) ** 2 + (zz - 0.55) ** 2) / 0.02) + 1500.0 * zz
nh = 3e16 * (1.0 + 2.0 * np.exp(-((xx - 0.6) ** 2 + (zz - 0.3) ** 2) / 0.03))
ne = 0.2 * nh
vturb = np.full_like(T, 5e3)
SC = {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 7, "formal_solver": "PiecewiseLinear",
      "interpolation": "Linear", "x_boundary": "Periodic"}
LONGER = {"quadrature": "StepanMirrored", "quadrature_order": 7, "x_boundary": "Periodic",
          "characteristics": "LongerCharacteristics"}


def timer(log, name):
    m = re.search(rf"^ *{re.escape(name)} +(\d+) +([0-9.e+-]+)", log, re.M)
    return (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)


def run_dexrt(exe, name, env=None, cmd_prefix=(), sc=SC):
    atmos_path = work / f"atmos_{n}.nc"
    if not atmos_path.exists():
        with ncdf.Dataset(atmos_path, "w") as f:
            f.createDimension("x", nx)
            f.createDimension("z", nz)
            for k, v in dict(temperature=T, ne=ne, nh_tot=nh, vturb=vturb, pressure=nh * KB * T,
                             vx=0 * T, vy=0 * T, vz=0 * T).items():
                f.createVariable(k, "f4", ("z", "x"))[...] = v
            f.createVariable("voxel_scale", "f4")[...] = dx
    conf = {
        "mode": "NonLte",
        "atmos_path": str(atmos_path),
        "output_path": str(work / f"{name}_out.nc"),
        "atoms": {"H": {"path": str(repo / "tests/H_6.yaml"), "treatment": "Active"}},
        "boundary_type": "Zero",
        "conserve_charge": False,
        "conserve_pressure": False,
        "initial_lambda_iterations": 0,
        "max_iter": 1,
        "ng_config": {"enable": False},
        "store_J_on_cpu": False,
        "system": {"mem_pool_gb": 6.0},
        "sc": sc,
    }
    conf_path = work / f"{name}.yaml"
    conf_path.write_text(yaml.dump(conf))
    log_path = work / f"{name}.log"
    with open(log_path, "w") as log:
        res = subprocess.run([*cmd_prefix, str(exe), "--config", str(conf_path)], cwd=work, stdout=log,
                             stderr=subprocess.STDOUT, env=env)
    text = log_path.read_text()
    if res.returncode != 0:
        print(text[-3000:])
        raise RuntimeError("dexrt_sc failed")
    with ncdf.Dataset(work / f"{name}_out.nc") as out:
        nwave = out["wavelength"].shape[0]
    ndir = int(re.search(r"(\d+) directions", text).group(1))
    calls, t_sweep = timer(text, "SC sweep")
    _, t_gamma = timer(text, "SC Gamma")
    _, t_mat = timer(text, "SC material")
    _, t_base = timer(text, "SC base emis/opac")
    _, t_j = timer(text, "SC J")
    _, t_total = timer(text, "DexRT SC")
    # NOTE: max_iter = 1 is two formal solutions (the second without Gamma).
    num_fs = 2
    nodes = nx * nz * nwave * ndir * num_fs
    return dict(nwave=nwave, ndir=ndir, nodes=nodes, sweep=t_sweep, gamma=t_gamma,
                material=t_mat + t_base, J=t_j, total=t_total, num_fs=num_fs)


def run_lw():
    pad = lambda a: np.concatenate([a, a[:, :1]], axis=1)
    flip = lambda a: np.ascontiguousarray(a[::-1].astype(np.float64))
    height = ((np.arange(nz) + 0.5) * dx)[::-1].copy()
    x = (np.arange(nx + 1) + 0.5) * dx
    atmos = lw.Atmosphere.make_2d(height=height, x=x, temperature=flip(pad(T)), vx=flip(pad(0 * T)),
                                  vz=flip(pad(0 * T)), vturb=flip(pad(vturb)), ne=flip(pad(ne)),
                                  nh_tot=flip(pad(nh)), z_lower_bc=lw.ZeroRadiation(),
                                  z_upper_bc=lw.ZeroRadiation())
    atmos.quadrature(3)
    a_set = lw.RadiativeSet([H_6_atom()])
    a_set.set_active("H")
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=lw_threads,
                     formal_solver="piecewise_linear_2d", interp_fn="interp_linear_2d")
    ctx.formal_sol_gamma_matrices()  # warm up
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        ctx.formal_sol_gamma_matrices()
        times.append(time.perf_counter() - t0)
    nwave = spect.wavelength.shape[0]
    ndir = 2 * atmos.Nrays  # each ray solved up and down
    nodes = (nx + 1) * nz * nwave * ndir
    return dict(nwave=nwave, ndir=ndir, nodes=nodes, total=min(times))


def ns(t, nodes):
    return 1e9 * t / nodes


rows = []
r = run_lw()
rows.append(("Lightweaver CPU {}t (FS+Gamma)".format(lw_threads), r, ns(r["total"], r["nodes"]), None))
for label, exe, env, prefix in [
    ("dexrt_sc GPU", dexrt_gpu, None, ()),
] + ([(f"dexrt_sc CPU {omp_threads}t", dexrt_cpu,
       {**os.environ, "OMP_NUM_THREADS": str(omp_threads), "OMP_PROC_BIND": "spread", "OMP_PLACES": "threads"},
       ("mpirun", "-n", "1", "--bind-to", "none"))] if dexrt_cpu else []):
    for kind, sc in [("SC", SC), ("Longer", LONGER)]:
        d = run_dexrt(exe, f"{label}_{kind}".replace(" ", "_"), env, prefix, sc)
        # NOTE: Gamma is only accumulated on one of the two formal solutions.
        per_fs_gamma = (d["sweep"] + d["material"] + d["J"]) / d["nodes"] + d["gamma"] / (d["nodes"] / d["num_fs"])
        rows.append((f"{label} {kind} sweep only", d, ns(d["sweep"], d["nodes"]), None))
        rows.append((f"{label} {kind} FS+Gamma", d, 1e9 * per_fs_gamma, None))

print(f"Grid {nx}x{nz}")
for label, d, t, _ in rows:
    print(f"{label:40s} nwave {d['nwave']:5d} ndir {d['ndir']:3d}  {t:8.3f} ns/(node dir wave)")
