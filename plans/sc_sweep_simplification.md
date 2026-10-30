# SC and Longer sweep simplification

## Objective

Make the implementation expose the numerical sequence directly: prepare a
batch, establish incoming intensity, advance through planes, record the
solution, and publish outgoing intensity. Reduce the total maintained code
and the number of supported internal paths. Moving existing verbosity into
new wrappers does not count as a reduction.

This is a refactor of the current numerical methods. Preserve Short and Longer,
2D and 3D, supported quadratures, fixed/periodic boundaries, non-uniform z,
formal solvers, interpolation choices, radiative losses, MPI, and PRD.

## Intended structure

    ScTaskExecutor
      prepare material and bind batch storage
      prepare boundary values
      run Short or Longer using a common launch policy
        initialise incoming characteristic state
        advance planes in dependency order
        record I_nl, b, q and optional losses
        retain outgoing characteristic state
      pack outgoing intensity and accumulate radiation

The dynamic scheduler continues to manage readiness, message ownership and
wavelength-slot admission. Numerical geometry and boundary choices stay in the
executor/sweep layer.

Four numerical implementations remain: Short 2D/3D and Longer 2D/3D. They share
setup and execution conventions where those conventions are actually the same.
In particular, Longer 2D gathers segment contributions into cells; Longer 3D
uses atomic accumulation. They do not become one generic numerical loop.

## 1. Establish the comparison baseline

Save the current source revision and CPU/CUDA executables before changing the
kernels. Record configurations and outputs for a small representative matrix:

- Short and Longer in 2D/3D, uniform and non-uniform z.
- Fixed, periodic and mixed lateral boundaries; physical and MPI inflow.
- Linear and limited interpolation; each supported Short formal solver.
- Optically thin/thick material, moving atmospheres and sparse/empty slabs.
- Radiative losses, background scattering and PRD.
- 3D with one team per direction/wavelength and with multiple plane chunks.
- Longer 2D with segment counts below and above SC_LONGER_SEG_BLOCK, including
  rays which wrap a periodic domain more than once.

Use the existing physical tests rather than introducing tests of wrapper
implementation details. Save timings, peak allocations and GPU register/spill
information for a few representative workloads. CUDA builds use 8 jobs.
GPU runtime comparisons require suitable hardware; compile success alone does
not justify removing performance specialisations.

Deliverable: reproducible comparisons, with existing numerical tolerances
written down. Do not loosen tolerances to accommodate the refactor.

## 2. Delete obsolete paths before adding abstractions

### Keep the three streams explicit

2D Healpix retains analytic along-y rays: rotation about y cannot remove
those rays. Preserve their solution and the LOCAL FIFO. UP and DOWN follow
opposite z dependencies; LOCAL has no MPI inflow and runs at low priority to
fill communication gaps. Stepan and 3D batches use UP/DOWN only.

Add a generated-quadrature test documenting this distinction. Do not remove
ScRayMode while it distinguishes a real numerical method. Simplify repeated
mode checks through prepared batch views and shared setup instead.

### Unused Longer traversal modes

The only current sc_lattice_rows call uses Gather=true and TrackTau=false.
Remove march-only and optical-depth-tracking paths, their template arguments,
unused tau buffers and stale comments about 3D in-plane consumers. Preserve
Blocked and Loss: both have active uses. Check all repository consumers again
before deletion.

Retire ScDagScheduler and its build entry after the saved baseline is usable.
If it is temporarily needed for comparison, keep it outside the maintained
production source rather than adapting two schedulers through every change.

Deliverable: a smaller implementation with fewer internal states, before any
kernel reorganisation. Run geometry/unit, transport and Longer tests here.

## 3. Prepare batch access once

Current Short kernels reconstruct material accessors, boundary lookups and
output indexing inside their launch bodies. Longer 2D separately assembles a
large lattice adapter. Replace repeated construction with small, concrete
views bound to one direction/wavelength pair:

- Material access: array pointers/strides and canonical-to-physical mapping,
  with existing halo, periodic and physical-edge rules.
- Boundary access: typed access to the prepared boundary storage.
- Solution storage: I_nl, b, q, incoming/outgoing intensity and optional loss.

Use dimension-specific access where necessary, sharing small indexing and
storage operations. Do not create a runtime-polymorphic accessor hierarchy or
one struct containing every optional buffer of every algorithm. Keep device
access inline and preserve restrict qualifiers where they matter.

The node solvers should receive these views directly. Their numerical entry
and segment-integration logic stays recognisable while the surrounding kernel
loses repeated pointer extraction, accessor definitions and store code.

Deliverable: implement on Short 2D first, then Short 3D. Apply only the proven
common pieces to Longer. Count net lines across all touched files, including
new headers; reject a change that merely relocates the boilerplate.

## 4. Consolidate batch, geometry and boundary preparation

Share direction grouping, chunking into SC_DIR_BATCH, device-array upload and
common row-step/path-length preparation between 2D and 3D. Retain the small
pieces which compute dimension-specific geometry, including Longer crossing
tables. Preserve global geometry decisions so all ranks produce matching jobs.

Give each algorithm a prepared boundary view with named inflow and side-face
access. Share boundary sampling/cone setup and coordinate operations where
identical. Keep Short ghost-node boundaries distinct from Longer lattice-entry
boundaries: their sampling locations are not interchangeable.

Replace duplicate solver/interpolation dispatch maps with one dispatch helper
used by the dimension-specific Short entry points. Keep algorithm selection
outside the inner numerical loops. Longer should not inherit irrelevant
Short interpolation/formal-solver machinery.

Deliverable: one place for common batch preparation and dispatch, with small,
explicit dimension/algorithm geometry functions. Run boundary, anisotropic,
stretched-z and decomposition comparisons before proceeding.

## 5. Share launch policy; simplify the numerical drivers

Make the following decisions in one launch-policy helper:

- Team sizing and limits.
- Whether a direction/wavelength pair owns an entire plane or several teams
  split it into chunks.
- Whole-slab versus one-plane-per-launch execution.

Keep dependency ordering explicit. One team can advance planes with team
barriers. Multiple teams require device-wide ordering between planes; a team
barrier cannot replace that ordering.

Use this policy in Short 3D and Longer 3D. Let each retain its numerical work
and incoming/outgoing state initialisation. Keep the 2D launch simpler where
it needs only one team per direction/wavelength pair; do not route it through
unused 3D branches.

Within each driver, aim for clearly visible operations:

    initialise incoming state
    for each assigned plane:
        solve nodes or march characteristic segments
        store/gather the cell solution and losses
        synchronise and advance characteristic state
    expose outgoing intensity

Fold the remaining 2D Longer traversal into its actual consumer if the former
generic adapter no longer serves a purpose. Keep its march/gather phases and
blocked/unblocked paths explicit. Preserve the different 3D atomic algorithm.

Deliverable: four short, readable drivers over shared access/preparation and
launch policy, with the numerical segment/node solvers still easy to locate.

## 6. Validate, measure and remove superseded code

After each stage, use its relevant existing regression tests. At the end run:

- tests/test_sc.cpp and the SC IO/layout checks.
- tests/sc/sc_dag_test.py (its Ng check is currently report-only).
- GivenFs transport, 3D versus 2D, path-length and prototype comparisons.
- tests/sc/sc_rad_loss_test.py and the PRD/scattering comparisons.
- CPU and CUDA compilation and representative old/new runtime measurements.

Compare physical outputs using existing tolerances; bitwise equivalence is
not required for changes in legal accumulation order. Investigate differences
rather than treating numerical tolerance as permission to change the method.

Measure GPU occupancy/registers/spills, kernel launches, scratch allocations,
transport overlap and end-to-end solve time on the same configurations. A
repeatable slowdown beyond run-to-run noise needs explanation and agreement;
retain a performance specialisation if its removal materially hurts a real
workload. Avoid timing builds and benchmarks concurrently.

Report net source reduction separately for obsolete scheduler deletion,
unsupported/unused paths and actual sweep simplification. The earlier
750–1,200-line sweep reduction is an investigation target, not a commitment.
Accept the final structure only if a reader can follow the plane dependencies,
boundary treatment and buffer lifetimes without traversing layers of wrappers.

## Review sequence

1. Baseline plus deletion of unused traversal modes; document the three streams.
2. Prepared access/storage and shared batching/boundary/dispatch preparation.
3. Shared launch policy, simplified numerical drivers and final measurements.

Keep these independently reviewable and numerically checked. Limit changes to
the sweep machinery and required callers; do not combine them with a new
formal solver, interpolation method, quadrature or storage precision.

## Implementation record

- Baseline: `5f34cbb`; saved CPU executables in
  `/tmp/sc_simplification_baseline`.
- Stage 1 preserves all three FIFOs at the user's request. Generated-quadrature
  testing confirms that folded Healpix order 0 includes AlongY rays. Removed
  the old scheduler and unused Gather/TrackTau traversal variants. Standalone
  tests: 27 cases / 713,990 assertions. Focused MPI suite passes in
  `/tmp/sc_stage1_regression`. No quadrature or scheduling policy changed.
- Stage 2 shares 2D/3D batch creation, Short variant dispatch, boundary sampling,
  and concrete Short storage views. CPU 2D/3D and CUDA builds pass; unit suite,
  focused MPI and two-rank 3D Longer tests pass. The existing 3D Short shift
  assertion fails at 0.0001497903 on both baseline and refactor; all four
  thin/thick/shift datasets are bitwise identical. Its 1e-5 threshold is left
  unchanged. Logs: `/tmp/sc_stage2_*`.

- Stage 3 uses one Short plane driver, a shared launch policy, and an explicit
  Longer 2D march/gather kernel. The single-consumer lattice adapter/header is
  removed. The original launch bounds, blocked Longer path and all three FIFOs
  remain. Boundary samplers borrow kernel state rather than copying array
  handles per entry. The 2D view retains its interior bounds check before any
  modulo and omits unused y-index work.

## Final validation and measured result

SC `.cpp`/`.hpp` source decreased from 10,219 to 9,110 lines: 1,109 removed,
including 741 from the retired scheduler and 368 from the active implementation.
Tests, documentation and the benchmark script are excluded. Active-code savings
are below the initial investigation target; numerical/performance distinctions
were retained rather than forcing further generalisation.

Checks completed:

- Final standalone suite: 27 cases / 713,990 assertions.
- Focused CPU MPI suite, including window/cache pressure, LOCAL Healpix,
  background scattering, empty ranks, conservation and restart. Its existing
  Ng check remains report-only.
- CPU and CUDA 2D/3D builds; CUDA builds used eight jobs.
- CPU and GPU 2D physical sweeps, including new analytic blocked/multi-lap
  Longer tests; CPU and GPU analytic 3D Longer tests.
- GPU 3D Short thin/thick, shift, flux/interpolation and Healpix checks.
- Baseline/current CPU comparisons for transport, NLTE, scattering, 3D and PRD;
  2D/3D radiative-loss conservation and wavelength-sum checks.
- GPU baseline/current thermal, scattering and PRD checks; a two-rank transport
  comparison with both ranks on the same GPU. Broader MPI checks ran on CPU.
- Known existing CPU test limitation: the strict 3D Short periodic-shift
  threshold fails on baseline and refactor at the same 0.0001497903 error;
  datasets are bitwise identical. The threshold was not changed. GPU Short
  shift checks pass. Nsight Systems captured CUDA API activity but no device
  kernel activity here, so no measured occupancy/overlap claim is made.

A transient boundary-handle-copy slowdown and unnecessary interior modulo were
caught by old/new timings and removed. Small ~30 ms GPU fixtures varied by
several milliseconds; the final GPU comparison uses larger workloads.

### Reproducible benchmark

`scripts/sc/bench_sweeps.py BASELINE_DIR CURRENT_DIR WORK [--gpu] [--repeats N]`
expects `dexrt_sc` and `dexrt_sc_3d` in both directories. Save baseline binaries
from commit `5f34cbb`. Run builds/tests separately from timed runs. The script
alternates baseline/current order, checks J/emergent output and quadrature
arrays on each repeat, and saves raw timings plus medians to `results.json`.
`SC_MPIEXEC` can override the MPI launcher.

GivenFs, Stepan order 7, periodic lateral boundaries, one rank and one CPU
thread. CPU fixtures: 128x128 with 16 wavelengths/B=4 and 32^3 with 8/B=2.
GPU fixtures: 512x512 with 128 wavelengths/B=8 and 96^3 with 24/B=3, on the
available RTX 4070. These are sweep fixtures, not atomic Ca II/NLTE or A100
scaling benchmarks. GPU timing is the rank-0 busy interval (including material
and accumulation); CPU timing is the complete `DexRT SC` interval.

### CPU (three repeats)

| Case | Baseline (s) | Current (s) | Change |
|---|---:|---:|---:|
| 2d_Short | 0.538228 | 0.449423 | -16.5% |
| 2d_LongerCharacteristics | 0.331135 | 0.324241 | -2.1% |
| 3d_Short | 1.396010 | 1.433132 | +2.7% |
| 3d_LongerCharacteristics | 0.370534 | 0.367751 | -0.8% |

### GPU (five repeats)

| Case | Baseline (s) | Current (s) | Change |
|---|---:|---:|---:|
| 2d_Short | 0.404200 | 0.405400 | +0.3% |
| 2d_LongerCharacteristics | 0.367400 | 0.367700 | +0.1% |
| 3d_Short | 0.427800 | 0.428600 | +0.2% |
| 3d_LongerCharacteristics | 0.267800 | 0.260600 | -2.7% |

The CPU 3D Short median is about 3% higher in this small fixture, with
baseline/current ranges overlapping; 2D Short is about 16% faster. Larger GPU
Short medians are within 0.3%, 2D Longer is unchanged, and 3D Longer is about
2.7% faster. These local measurements do not establish production speedups.

Static CUDA resource inspection for default 3D Short retains 64 registers and
the original launch bounds; stack allocation is 160 versus 128 bytes/thread.
No additional device scratch arrays were introduced. A100 runtime/resource
validation remains a hardware-specific follow-up.

Raw outputs/logs: `/tmp/sc_stage3_*`, `/tmp/sc_final_gpu_*`,
`/tmp/sc_sweep_final_cpu`, `/tmp/sc_sweep_final_gpu_large`. The isolated CUDA
reference rebuilt the baseline main/sweep objects with the same compiler flags
and reused unchanged common runtime objects. No profiler-derived device-kernel
timings are substituted for the documented host busy intervals.
