"""ScIo: dexrt_sc_io_test's round trip on 1-4 ranks with both backends, the
files identical between backends, and rank 0's streaming memory independent
of the block size (Gather).
Usage: python sc_io_test.py DEXRT_SC_IO_TEST WORKDIR"""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import netCDF4 as ncdf

exe = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
work.mkdir(parents=True, exist_ok=True)
mpi = shlex.split(os.environ.get(
    "SC_MPIEXEC",
    "mpirun --oversubscribe --mca pml ob1 --mca btl self,tcp --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0",
))
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def run(ranks, *args):
    res = subprocess.run(mpi + ["-n", str(ranks), str(exe), *map(str, args)], capture_output=True, text=True)
    return res.returncode, res.stdout


def same(a, b):
    with ncdf.Dataset(a) as x, ncdf.Dataset(b) as y:
        return (
            set(x.variables) == set(y.variables)
            and all(np.array_equal(x[k][...], y[k][...]) for k in x.variables)
            and all(np.array_equal(np.asarray(x.getncattr(k)), np.asarray(y.getncattr(k))) for k in x.ncattrs())
        )


for ranks in [1, 2, 3, 4]:
    for backend in ["Gather", "Parallel"]:
        code, out = run(ranks, work / f"io_{ranks}_{backend}.nc", backend)
        check(code == 0, f"round trip {ranks} ranks {backend}")
    check(same(work / f"io_{ranks}_Gather.nc", work / f"io_{ranks}_Parallel.nc"), f"{ranks} ranks: backends write identical files")

growth = {}
# NOTE: Blocks of several 256 MB chunks: rank 0 holds at most two chunks.
for mb in [512, 2048]:
    code, out = run(3, work / "big.nc", "Gather", mb)
    check(code == 0, f"streaming {mb} MB/rank")
    growth[mb] = [float(m) for m in re.findall(r"rank 0 peak RSS growth: write (\d+) MB", out)]
(work / "big.nc").unlink(missing_ok=True)
check(growth[512] and growth[2048] and growth[2048][0] <= growth[512][0] * 1.1 + 16 and growth[2048][0] <= 2 * 256 + 64,
      f"rank 0 streaming memory independent of block size ({growth[512]} vs {growth[2048]} MB)")

if failures:
    print(f"{len(failures)} failures")
    sys.exit(1)
print("All ScIo tests passed")
