"""Compare saved baseline/current SC binaries without overlapping compilation.

Usage: python scripts/sc/bench_sweeps.py BASELINE_DIR CURRENT_DIR WORK [--gpu]
Both directories contain dexrt_sc and dexrt_sc_3d. GPU timing uses the rank-0
busy interval (material, sweep, accumulation); CPU timing uses DexRT SC.
SC_MPIEXEC can override the one-rank MPI launcher. Results include raw times.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

import netCDF4 as nc
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/sc"))
from sc_givenfs_tests_lib import write_atmos as write2
from sc_3d_givenfs_tests import write_atmos as write3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    binaries = {key: getattr(args, key).resolve() for key in ("baseline", "current")}
    mpi = shlex.split(os.environ.get("SC_MPIEXEC", "mpirun --oversubscribe --mca pml ob1 --mca btl self,tcp --mca coll_hcoll_enable 0 --mca coll_ucc_enable 0"))
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    shapes = [(2, (512, 512, 128), 8), (3, (96, 96, 96, 24), 3)] if args.gpu else [(2, (128, 128, 16), 4), (3, (32, 32, 32, 8), 2)]
    metric = "rank0_busy" if args.gpu else "solver"
    pattern = r"SC dynamic:.*?busy ([\d.]+)s" if args.gpu else r"^\s*DexRT SC\s+\d+\s+([\deE.+-]+)"
    results = []
    for dim, shape, batch in shapes:
        coords = np.indices(shape[:-1])
        blob = np.exp(-sum(((c - n / 2) / (n / 5)) ** 2 for c, n in zip(coords, shape[:-1])))
        chi = np.broadcast_to((0.1 + blob)[..., None], shape).astype("f4").copy()
        eta = (chi * (1 + blob[..., None])).astype("f4")
        atmos = work / f"atmos{dim}.nc"
        (write2 if dim == 2 else write3)(atmos, eta, chi)
        for method in ("Short", "LongerCharacteristics"):
            name = f"{dim}d_{method}"
            times = {key: [] for key in binaries}
            outputs = {}
            for repeat in range(args.repeats):
                order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                for variant in order:
                    tag = f"{name}_{variant}_{repeat}"
                    output = work / f"{name}_{variant}.nc"
                    config = work / f"{tag}.yaml"
                    config.write_text(yaml.safe_dump({
                        "mode": "GivenFs", "atmos_path": str(atmos), "output_path": str(output),
                        "system": {"mem_pool_gb": 2},
                        "sc": {"characteristics": method, "quadrature_order": 7, "wave_batch": batch,
                               "x_boundary": "Periodic", "y_boundary": "Periodic"},
                    }))
                    binary = binaries[variant] / ("dexrt_sc" if dim == 2 else "dexrt_sc_3d")
                    log = work / f"{tag}.log"
                    with log.open("w") as stream:
                        subprocess.run([*mpi, "-n", "1", str(binary), "--config", str(config)],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
                    match = re.search(pattern, log.read_text(), re.M)
                    if not match:
                        raise RuntimeError(f"Missing {metric} timer in {log}")
                    times[variant].append(float(match[1]))
                    outputs[variant] = output
                with nc.Dataset(outputs["baseline"]) as a, nc.Dataset(outputs["current"]) as b:
                    for key in ("J", "I_emergent", "sc_mu", "sc_wmu"):
                        if key in a.variables:
                            np.testing.assert_allclose(a[key][...], b[key][...], rtol=1e-3 if key == "J" else 1e-5,
                                                       atol=2**-48 if key == "J" else 1e-25, err_msg=f"{name}: {key}")
            record = {"case": name, "metric": metric,
                      "median": {key: statistics.median(values) for key, values in times.items()}, "raw": times}
            results.append(record)
            print(json.dumps(record), flush=True)
    (work / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
