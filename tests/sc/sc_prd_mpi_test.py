"""PRD on multiple ranks: H_4 (test_atom, Lyman alpha in PRD) on an atmosphere made by
lw_compare_prd.py, a fixed number of iterations on 1, 2 and 3 ranks. J*
accumulates in a rank-count dependent order, so agreement is to rounding.
Usage: python sc_prd_mpi_test.py DEXRT_SC WORKDIR ATMOS_NC"""
import os
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf
import yaml

dexrt_sc = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
atmos = Path(sys.argv[3]).resolve()
work.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
mpi = shlex.split(os.environ.get(
    "SC_MPIEXEC",
    "mpirun --oversubscribe --mca pml ob1 --mca btl self,tcp --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0",
))


def run(ranks):
    name = f"prd_{ranks}"
    conf = {
        "mode": "NonLte",
        "atmos_path": str(atmos),
        "output_path": str(work / f"{name}.nc"),
        "atoms": {"H": {"path": str(repo / "tests/test_atom.yaml"), "treatment": "Active"}},
        "boundary_type": "Thermalised",
        "conserve_charge": False,
        "max_iter": 12,
        "pop_tol": 1e-12,
        "ng_config": {"enable": False},
        "system": {"mem_pool_gb": 1.0},
        "sc": {"characteristics": "Short", "quadrature": "StepanMirrored", "quadrature_order": 3, "interpolation": "Linear",
               "x_boundary": "Periodic", "wave_batch": 64, "prd": "Hybrid"},
    }
    (work / f"{name}.yaml").write_text(yaml.dump(conf))
    env = dict(os.environ, OMP_NUM_THREADS="2", OMP_PROC_BIND="false")
    with open(work / f"{name}.log", "w") as log:
        subprocess.run(mpi + ["-n", str(ranks), str(dexrt_sc), "--config", str(work / f"{name}.yaml")],
                       cwd=work, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=1800)
    out = ncdf.Dataset(work / f"{name}.nc")
    return np.asarray(out["pops"][...]), np.asarray(out["prd_rho"][...]) if "prd_rho" in out.variables else None


ref_pops, ref_rho = run(1)
assert ref_rho is not None, "no PRD output"
worst = 0.0
for ranks in [2, 3]:
    pops, rho = run(ranks)
    dp = np.max(np.abs(pops - ref_pops) / np.abs(ref_pops))
    dr = np.max(np.abs(rho - ref_rho)) if rho is not None else np.inf
    print(f"{ranks} ranks: max rel pops diff {dp:.2e}, max |drho| {dr:.2e}")
    worst = max(worst, dp / 2e-3, dr / 5e-3)
# NOTE: rho* and J* are stored in 16 bits (rho ~ 1: steps of 2^-10) and
# re-rounded each iteration; the wavelength batches (so f32 precon_R order)
# depend on the rank count.
if worst > 1.0:
    raise SystemExit("PRD MPI results differ")
print("PRD MPI test passed")
