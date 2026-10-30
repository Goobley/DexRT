"""Check per-rank scheduler reports and numerical results with diagnostics on/off.

Usage: python sc_scheduler_stats_test.py DEXRT_SC WORKDIR [--launcher 'mpirun ...']
Use --dim 3 for dexrt_sc_3d and --allow-roundoff for GPU Longer atomic sums.
Runs one and four ranks with OMP_NUM_THREADS=1.
"""
import argparse
import copy
import os
from pathlib import Path
import re
import shlex
import subprocess

import netCDF4 as nc
import numpy as np
import yaml


def check_report(log, ranks):
    """Reports must cover each rank once, with consistent work and wait totals."""
    blocks = re.split(r"  SC scheduler rank (\d+): ", log)
    assert [int(rank) for rank in blocks[1::2]] == list(range(ranks))
    for rank, block in zip(blocks[1::2], blocks[2::2]):
        total = int(re.search(r"active_cells=\d+ tasks=(\d+)", block)[1])
        streams = re.findall(r"stream \w+: tasks=(\d+) busy=([\d.]+)", block)
        assert sum(int(count) for count, _ in streams) == total
        header = dict(re.findall(r"(\w+)=([\d.]+)", block.splitlines()[0]))
        assert all(float(value) >= 0 for value in header.values())
        assert abs(sum(float(time) for _, time in streams) - float(header["busy"])) < 3e-6
        waits = re.findall(r"wait [\w+]+: calls=\d+ seconds=([\d.]+)", block)
        assert abs(sum(map(float, waits)) - float(header["wait"])) < 8e-6
        accounted = sum(
            float(header[key]) for key in ("busy", "wait", "drain", "setup", "finish", "overhead")
        )
        assert abs(accounted - float(header["elapsed"])) < 5e-6
        assert re.search(r"cache: entries=2 builds=[1-9]\d* hits=\d+", block), rank
        phase_line = re.search(r"phases: (.*)", block)[1]
        phases = dict((key, float(value)) for key, value in re.findall(r"(\w+)=([\d.]+)", phase_line))
        assert phases["coefficients"] > 0 and phases["sweep"] > 0
        assert sum(phases.values()) <= float(header["busy"]) + 1e-5


def main():
    """Compare real solves with diagnostics disabled/enabled on one and four ranks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--characteristics",
        choices=("Short", "LongerCharacteristics"),
        default="LongerCharacteristics",
    )
    parser.add_argument(
        "--allow-roundoff",
        action="store_true",
        help="Use existing tolerances for nondeterministic GPU atomic sums",
    )
    parser.add_argument(
        "--launcher",
        default=(
            "mpirun --oversubscribe --mca pml ob1 --mca btl self,tcp "
            "--mca coll_hcoll_enable 0 --mca coll_ucc_enable 0"
        ),
    )
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    shape = (64, 16) if args.dim == 2 else (64, 16, 16)
    dims = ("z", "x") if args.dim == 2 else ("z", "y", "x")
    atmos = work / "atmos.nc"
    with nc.Dataset(atmos, "w") as f:
        for name, size in zip(dims, shape):
            f.createDimension(name, size)
        for key, value in dict(
            temperature=6000,
            ne=6e15,
            nh_tot=3e16,
            pressure=3e16 * 1.380649e-23 * 6000,
            vturb=5000,
            vx=0,
            vy=0,
            vz=0,
        ).items():
            f.createVariable(key, "f4", dims)[:] = value
        f.createVariable("voxel_scale", "f4")[:] = 1e5
    config = dict(
        mode="NonLte",
        atmos_path=str(atmos),
        atoms={"H": dict(path=str(repo / "tests/H_4.yaml"), treatment="Active")},
        boundary_type="Zero",
        max_iter=1,
        final_dense_fs=False,
        ng_config=dict(enable=False),
        system=dict(mem_pool_gb=0.125),
        output=dict(lte_pops=False, sc_emergent=True),
        sc=dict(
            characteristics=args.characteristics,
            wave_batch=7,
            base_slots=2,
            receive_window=2,
        ),
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    for ranks in (1, 4):
        outputs = []
        for detailed in (False, True):
            name = f"r{ranks}_{'detailed' if detailed else 'default'}"
            cfg = copy.deepcopy(config)
            if detailed:
                cfg["sc"]["scheduler_stats"] = True
            out = work / f"{name}.nc"
            cfg["output_path"] = str(out)
            path = work / f"{name}.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with (work / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        *shlex.split(args.launcher),
                        "-np",
                        str(ranks),
                        str(args.binary.resolve()),
                        "--config",
                        str(path),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            text = (work / f"{name}.log").read_text()
            if detailed:
                check_report(text, ranks)
            else:
                assert "SC scheduler rank" not in text
            outputs.append(out)
        with nc.Dataset(outputs[0]) as a, nc.Dataset(outputs[1]) as b:
            for key in ("pops", "J", "I_emergent"):
                x, y = np.asarray(a[key][:]), np.asarray(b[key][:])
                if ranks == 1 and not args.allow_roundoff:
                    assert x.tobytes() == y.tobytes(), key
                else:
                    atol = 2.0**-48 if key == "J" else max(1e-25, 1e-10 * np.max(np.abs(x)))
                    np.testing.assert_allclose(x, y, rtol=1e-3 if key == "J" else 2e-5, atol=atol)
        print(f"PASS scheduler stats: {ranks} ranks, reports and numerical agreement", flush=True)

    # A mixed flag must fail collectively before ranks can enter different report collectives.
    command = shlex.split(args.launcher)
    for rank in (0, 1):
        cfg = copy.deepcopy(config)
        cfg["sc"]["scheduler_stats"] = rank == 0
        cfg["output_path"] = str(work / "mismatched.nc")
        path = work / f"mismatched_{rank}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        if rank:
            command.append(":")
        command += ["-np", "1", str(args.binary.resolve()), "--config", str(path)]
    with (work / "mismatched.log").open("w") as log:
        result = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    assert result.returncode != 0
    rejection = (work / "mismatched.log").read_text()
    assert "SC ranks disagree on task boundaries or stream membership" in rejection
    print("PASS scheduler stats: mixed flags rejected collectively", flush=True)


if __name__ == "__main__":
    main()
