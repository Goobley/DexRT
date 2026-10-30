# 3D in dexrt_sc (plan for critique, 2026-10-25)

## Goal
- Run the SC solver on 3D atmospheres, keeping everything else intact:
  - the z-slab MPI decomposition and the DAG scheduler;
  - non-uniform z;
  - the formal solvers, the PRD treatment and the physics.
- The x/y chunk decomposition is later work. The 3D kernels are written so
  that chunking adds only exchanges.
- Most of the change is mechanical: rows become planes, columns become
  (x, y) nodes, and `Coord2` becomes `Coord3`.
- The real new pieces are:
  1. the upwind trace through the x and y face crossings;
  2. the interpolation on a plane or face, which is two-dimensional. The
     limited schemes need an unsplit reconstruction with a cross term and a
     joint limiter.

## What carries over unchanged
- **Per-cell work:** stat_eq, NR, the conservation steps, Ng, collisions, and
  background opacity (J_prev). They index by ks only.
- **PRD:** the J* gather, the redistribution and ρ_lab. The shift v·μ is
  already the full 3-vector.
- **Quadratures:** full sphere, 3-vectors. In 3D, `fold_y` never applies.
- **Formal step:** `sc_formal_step` (all three solvers) and the f32
  thresholds.
- **Boundaries:** `sample_boundary(bc, la, pos, mu)` for the Zero,
  Promweaver and Thermalised conditions, and the Promweaver cone. They
  already take a vec3 position and direction.
- **Per-plane z tables:** `row_t` and `row_ds` become `plane_tx`,
  `plane_ty` and `plane_ds`.
- **DAG scheduler:** tasks, streams, rings, base cache and progress logic.
  The upstream "row" becomes a plane of Nx·Ny values per (dir, wave).

## Geometry (`ScGeometry3d.hpp`)
- **Grid.** `ScGrid3d` holds num_x, num_y, num_z, dx, dy, dz, the offsets
  and `z_node`. x and y are uniform; z may be non-uniform.
  - Read `voxel_scale_y` (falling back to `voxel_scale`) and the new option
    `sc.y_boundary: Fixed | Periodic`, which mirrors `x_boundary`.
- **Frame.** `ScFrame3d` has flip_x, flip_y and flip_z, so the ray moves
  towards +s_x, +s_y and +p.
  - Per plane step Δz_p: t_x = |μx|Δz_p/(|μz|dx), t_y = |μy|Δz_p/(|μz|dy),
    ds = Δz_p/|μz|.
  - Direction batches group by flip_z only, as now. The x and y flips are
    per direction.
- **Upwind trace for node (p, i, j).**
  - The upwind point on plane p−1 is (i − t_x, j − t_y).
  - Between that point and the node, the characteristic crosses the node
    planes x = c (for c in (i − t_x, i)) and y = c' (for c' in (j − t_y, j)).
  - Each crossing has a height h ∈ (0, 1), measured as a fraction of the step
    from plane p−1:
    - h = (c − x0)/t_x for an x crossing;
    - h = (c' − y0)/t_y for a y crossing.
  - The two crossing sequences are merged in increasing h, with a two-pointer
    DDA.
  - At each crossing, the material is interpolated on that face:
    - on an x face, in (y, z), at (y0 + h·t_y, p − 1 + h);
    - on a y face, in (x, z).
  - The intensity is interpolated only on plane p−1, as in 2D. Formal steps
    integrate forward through the crossings.
  - Cost: t_x + t_y crossings per node, about 2× the 2D cost per crossing.
    The worst Stepan ray is il15n11 at θ = 86°, which crosses about 14
    planes per step for cubic cells.
- **Edges.**
  - With fixed x or y, the characteristic enters through whichever side face
    it crosses last on the way back (the larger h). The boundary value is
    sampled there.
  - With periodic axes, the indices wrap. Edges and corners follow the 2D
    side and edge logic per axis.
  - The z inflow face (half a step below plane 0) is unchanged.
- **Ray modes.** There are two: Sweep, and InPlane, which replaces
  Horizontal. AlongY disappears.
  - **When a ray is InPlane.** μz ≈ 0, or the same per-plane criterion as
    2D Horizontal: periodic and more wraps than `SC_MAX_PERIODIC_WRAPS`. A
    constexpr cap on t_x + t_y crossings per plane step also qualifies, so
    that rotated near-equatorial rays stay affordable.
  - **How it is solved.** Each z plane is a contiguous [y, x] block, i.e. a
    2D (rows = y, columns = x) problem, so it is solved by the 2D SC solver
    through its accessors:
    - the primary axis is whichever of x and y has the larger |μ_a|/Δa;
    - the transpose goes through the canonical index;
    - if the ray runs along the other axis, the 2D Horizontal row solver is
      used.
    - The ray's z drift is ignored, as 2D does for Horizontal.
  - **Open problem: the seam.** The 2D solver's primary axis has an inflow
    face. A periodic primary axis needs a seam treatment (lagged seam
    passes, as in the prototype's `run_passes`, or a fixed point). Decide
    when building it.
  - **Healpix stays.** It is rotated slightly, so no pixel has μz = 0
    exactly, and its near-equatorial rays go to InPlane. Stepan has no
    InPlane rays; its worst is θ = 86°.

## Interpolation on a plane or face (`ScInterp2d.hpp`)
Successive 1D finite-volume remaps, the tensor product of the existing
`sc_interp`, as in IRIS (Ibgui et al. 2013). IRIS does successive monotone
cubic point interpolation; ours is the conservative remap.

    interp_2d(F, tx, ty):            // F[i][j], i, j in {-2..1}
        g[j] = sc_interp<Interp>(F[-2..1][j], tx)   for j in -2..1
        return sc_interp<Interp>(g[-2..1], ty)

- **As a finite-volume average.** g[j] is the average of row j over the x
  window, i.e. a strip average. The y pass remaps the strip averages, so
  the result is the window average of a consistent reconstruction.
- **Cross term.** It is included: unlimited, this is the tensor-product
  operator, which reproduces f_xy.
- **Order of accuracy.** Second order. The 2D error is
  (Ax − Ex)Ay + Ex(Ay − Ey).
- **Bounds.** Each pass is bounded and positivity preserving for
  t ∈ [0, 1], so the composition is too. The result stays within the 4×4
  extremes, and no joint limiter is needed.
- **Linear and 1D limits.** Linear gives exactly bilinear. y-invariant data,
  or ty = 0, gives the current 1D result exactly.
- **Order of the passes.** It matters wherever a limiter is active, because
  the y limiter sees x-remapped values.
  - Measure the azimuthal bias first.
  - If needed, average the two orders, at 10 calls instead of 5.
- **Face crossings.** The axes are (horizontal, z): the horizontal pass at
  the fractional position, then z at 1 − h, in index space.
- **Cost.** 16 reads and 5 calls to `sc_interp` per field.
- **Comparator only (trial):** an unsplit reconstruction, with limited sx
  and sy, a limited twist sxy, and a Barth–Jespersen corner bound (twist
  clipped first).

## Kernel (`ScSweepNode3d.hpp`, `ScSweep3d.cpp`)
- **`sc_solve_node_3d<Solver, Interp>`** has the same accessor style as 2D:
  - `fetch_mat(p, i, j)`;
  - `fetch_I(p−1, i, j)`;
  - `bc` lookups for the bottom face, the side entries and the ghost frame.
- **Dispatch** is the same 9-entry map over (solver × interp).
- **Launch: one kernel per plane**, flat over (dir_in_batch, wave, j, i).
  - The 2D kernel uses one team per (dir, wave), with a team barrier per row.
    In 3D that underfills the GPU: a 256² plane gives 65k nodes, while memory
    limits the batch to tens of (dir, wave) pairs.
  - Per plane there are millions of independent nodes, so Nz launches per
    task is cheap (about 230k launches, roughly 1 s, against about 60 s of
    sweep per formal solution at 256³ × 24 dirs × 900 waves).
  - On CPU the same per-plane loop runs over (dir, wave, j), vectorised over
    i.
- **Boundary values (`ScBcLayout3d`)**, per (dir, wave):
  - the inflow z face, Nx·Ny entries;
  - per plane, a ghost frame 2 nodes wide on the inflow x and y sides, for
    the I stencils;
  - side-entry samples for nodes whose characteristic enters through an x or
    y side face, sized by `side_max_x/y` as in 2D.
  - These are precomputed by a pre-pass, so the kernel carries no boundary
    code.
- **Workspace.** Volume buffers as now: [dir, wave, (p + halo)·Nx·Ny + …]
  for chi and S (2 halo planes each side), and I_nl, b, q.
  - The memory chooser gains Ny, and gains dir sub-batching when even one
    wavelength does not fit. The dir batch size is then derived from memory,
    not tuned.
  - Cost: about 5 floats per cell per (dir, wave), i.e. 336 MB at 256³. A
    12 GB GPU holds about 24 pairs: a full 12-direction batch at 2 waves.
    z-slabs divide this by the rank count.
  - Rolling planes would remove the Nz factor, but they are a restructure:
    fill, sweep and accumulate per plane inside the plane loop. That is Q2.

## Infrastructure (the mechanical part)
- **`ScActiveMap`**: a mask over [z, y, x], `num_y`, and `Coord3 cell_of`.
  - `ScRunPiece` gains y. Runs are maximal x runs per (z, y).
  - ks is in row-major (z, y, x) order.
  - In 2D, `num_y = 1`, so the 2D path's arithmetic is unchanged.
  - 3D tiles use the dexrt_3d criterion on aligned global `BLOCK_SIZE_3D`³
    tiles. The `BLOCK_SIZE²` static_assert in `ScRadiation.cpp` is
    generalised.
- **Canonical indexing:** `sc_canonical_idx` and `sc_physical_coord` in 3D.
  This covers the material fill, the Gamma/J/PRD J* accumulation (one thread
  per run piece, unchanged), the emergent intensity and the clearing of
  inactive cells.
- **`StateSetup`:** the 3D atmosphere load through the core
  `load_atmos_3d_host`, then sparsification with the SC mask.
  - Restarts, snapshots and the compact layout gain y.
  - GivenFs inputs gain a 3D layout, [z, y, x, wave].
- **Decomposition, ghosts and pipeline:**
  - `sc_decompose_z` weights by active cells per plane.
  - `ScGhostInputs`: rows become planes. The packed fields are unchanged.
  - The upstream I message is one plane per (dir, wave): about 6 MB at 256²
    for 12 dirs × 2 waves.
  - The layout check in the DAG description gains Ny and the y boundary.
- **Output:**
  - Dense and sparse variables gain a y dimension. The active mask is
    [z, y, x].
  - The emergent intensity [la, dir, y, x] is 5.6 GB at 256² × 24 × 900, so
    in 3D it is opt-in (`output.sc_emergent`) and held on the host.
  - J output at 256³ × 900 waves is about 60 GB dense. Dense J output in 3D
    gets a warning in the log, and the existing `output.J` is the switch.
- **Executable:** a new `dexrt_sc_3d` target, as dexrt_3d is. It shares the
  SC sources, and the dimension-specific files are 2d/3d pairs. The 2D path
  is untouched. LongerCharacteristics is not built into the 3D target.
- **dexrt_py:**
  - the 3D atmosphere and GivenFs writers;
  - `y_boundary` in the schema;
  - plot helpers for (x, y) slices.

## Stages and validation (reordered 2026-10-25: Longer first)
- **T1, shared 3D infrastructure, with no physics change.**
  - Work:
    - the grid gains y, i.e. num_y and dy (`ScGrid`);
    - the frame gains flip_y and num_sy; the canonical index is
      (p·num_sy + sy)·num_s + s, which in 2D is the same index;
    - the active map, run pieces and `cell_of` move to (x, y, z);
    - the per-cell kernels (fill, Gamma, J, J*, emergent, pack) and the
      ghost inputs work on planes of num_x·num_y;
    - the 3D atmosphere and GivenFs load;
    - IO;
    - the `dexrt_sc_3d` target.
  - Check: 2D output is bitwise unchanged (Snow KHI benchmark, DAG suite,
    PRD MPI test).
- **T2, 3D LongerCharacteristics** (constant material only, as in 2D: the
  limited-linear material was dropped there as not robust in practice).
  - Tables: per (dir, plane), the merged x/y crossing pattern, giving the
    segment lengths and (ox, oy) cell offsets.
  - Kernel: per-plane phase A (march), then phase B (gather).
  - Boundaries: the inflow z face, the x/y side faces, and periodic wraps.
  - Unit tests:
    - per-cell path sums equal H/|μz|;
    - every cell is covered.
  - GivenFs tests:
    - thin emission is exact;
    - in a thick medium J tends to S;
    - the searchlight width stays at the source;
    - periodic shifts.
- **T3, 3D SC.**
  - The upwind trace through the merged crossings.
  - `sc_interp_2d` on the plane and on the faces (done; see the trial
    results).
  - `ScBcLayout3d`, a per-plane kernel, and the 9-variant dispatch.
  - The same GivenFs suite, plus the searchlight azimuth study.
- **T4, non-LTE** (both characteristics).
  - 1D equivalence: FAL-C tiled in x and y against Lightweaver 1D.
  - 2D equivalence: a y-invariant 3D atmosphere against 2D dexrt_sc.
  - Against dexrt_3d on a 3D cube.
  - PRD on a 3D shear.
- **T5, InPlane rays** (Healpix, rotated).
  - The 2D solvers (SC and longer) on contiguous z planes.
  - A seam treatment for periodic primary axes.
- **T6, MPI.** 1, 2 and 4 ranks agree with 1 rank to rounding (uniform and
  stretched z).
- **T7, performance.** 128³ and 256³ GPU runs, both characteristics.

## Later (not in this plan)
- **x/y chunk decomposition.** This is a KBA-style pipeline: the DAG streams
  become (z sweep × x/y quadrant). The upwind trace reaches up to ⌈t⌉ + 2
  nodes into the neighbouring chunk on plane p−1, so the halo width depends
  on the direction. That is the main design question there.
- **Rolling-plane fused tasks**, if Q2 is deferred.
- **InPlane seams** for periodic primary axes, if v1 defers them.
- **LongerCharacteristics in 3D** (a 2D lattice of characteristics).
- **Non-uniform x and y.**

## Decisions (2026-10-25, user)
1. Interpolation: successive 1D limited remaps (IRIS-style tensor product).
   Trial first.
2. Memory: volume buffers for now.
3. Healpix stays: rotated, with the InPlane mode handled by a separate kernel
   that reuses the 2D solver on the contiguous x–y planes.
4. A new executable, `dexrt_sc_3d`.
5. `sc.y_boundary` and `voxel_scale_y`, mirroring 2D.
6. The 3D emergent intensity is opt-in.

## Trial results (2026-10-25)
Script: `scripts/sc/interp2d_trial.py`, a numpy port of `sc_interp` in a
vacuum plane-remap loop. Header: `source/ScInterp2d.hpp`. Unit test:
`SC plane interpolation`.

- **Bounds.**
  - Variants: successive xy, yx and symmetrised, plus the unsplit BJ
    version.
  - Data: a top-hat, a diagonal step and a checkerboard.
  - Every variant stays within [0, 1] with no negatives.
- **Order of accuracy.**
  - Data: a smooth, non-separable diagonal ellipse, carried a fixed
    distance, N = 32..256.
  - VanLeer gives 1.87 and MC gives 2.03 for all orders (Linear 0.8). The
    split and unsplit versions are equally accurate.
- **Order dependence.** For |xy − yx|/max after 48 planes, it is 2e-4 (disc)
  and 1e-3 (ellipse).
- **Azimuthal symmetry.**
  - Data: a disc beam, θ = 60°, 48 planes, φ = 0..90°.
  - Single order: the φ ↔ 90° − φ asymmetry in added variance is 1.5e-3
    cells², against an added variance of about 2 cells² (under 0.1%).
    The ellipse tilt Vxy is under 3e-4.
  - Symmetrised: the asymmetry is at rounding.
- **Spread.** Mean added variance: MC split 2.04, MC unsplit 2.38, VanLeer
  3.05, Linear 17.2.
  - The unsplit joint limiter is more diffusive.
  - It also doesn't reduce to 1D at tb = 0, because φ uses sy.
  - The spread against azimuth is set by the fractional Courant numbers, as
    in 1D.
- **Conclusion.** Successive 1D remaps with a single, fixed order (a then b).
  No symmetrisation, and the unsplit variant is dropped.

## Status (2026-10-25)
- **T1 done.**
  - `ScGrid` and `ScFrame` now carry y, and `ScGeometry.hpp` holds the
    shared types.
  - The active map works in planes, with `Coord3` cells and runs that carry
    y.
  - Kernels, ghosts, IO, restarts and the 3D atmosphere/GivenFs loads are
    plane-generic.
  - Targets: the `dexrt_sc_common` OBJECT library, plus `dexrt_sc` and
    `dexrt_sc_3d` (`ScMain3d.cpp` defines `DEXRT_SC_3D`).
  - The 11 2D regression cases (scratchpad `t1/run_cases.py`) are bitwise
    identical to HEAD: GPU and CPU, MPI with 2 ranks, PRD, Longer, and
    non-uniform z.
- **T2 done (GivenFs).**
  - Code: `ScGeometry3d.hpp` (frames, positions, `sc_longer_plane`, side
    index), `ScSweep3d.cpp` (batches, tables) and `ScLongerSweep3d.cpp`.
  - `tests/sc/sc_3d_givenfs_tests.py`:
    - thin emission: exact to 2e-7;
    - thick medium: J = S;
    - periodic x/y shift: bitwise;
    - emitting box in vacuum, each lattice line's emergent I against η ×
      its analytic path: 1-3e-7, on cubic, anisotropic (1.3/0.7/0.9) and
      stretched-z grids, and with il15.
  - MPI: 1, 2 and 3 ranks are bitwise identical, except il15 (1e-7, from
    the J accumulation order over direction batches). GPU vs CPU: 1e-7.
  - Unit test "SC longer characteristics 3D plane tables": segment cells,
    path sums, the 2D-row limit, and the side index bijection.
  - Not yet in 3D:
    - Thermalised and Promweaver side/bottom boundaries (only Zero is
      tested);
    - InPlane rays (they throw);
    - SC (it throws).
- **T3 done (GivenFs).**
  - Code: `ScSweepNode3d.hpp` (upwind trace, x/y crossings merged by
    height, `sc_interp_2d` on plane p−1 and on the crossed faces,
    edge-half-cell face values as in 2D) and `ScBcLayout3d`. The kernel and
    the 9-variant dispatch are in `ScSweep3d.cpp`.
  - `sc_3d_givenfs_tests.py --characteristics Short`:
    - thin and thick are exact; periodic shifts agree to 1.5e-7;
    - emergent flux is conserved to 3e-6;
    - added beam variance against the exact longer result, over 24 planes:
      Linear 9.0, VanLeer 2.6, MC 2.1 cells².
  - MPI: 3 ranks are bitwise identical to 1 rank.
  - `tests/sc/sc_3d_vs_2d.py` (y-invariant, periodic y): longer agrees to
    1.8e-7, SC to 1.5e-3 (truncation).
- **T4, first checks.** The Thermalised bottom now takes a plane.
  `tests/sc/sc_3d_falc.py`: H_6 on FAL-C at its native heights,
  horizontally homogeneous, 3D against 2D.
  - **Longer:** LTE J agrees to 8e-7. Non-LTE pops: median 2.7e-5, max
    4.4e-4 (convergence level).
  - **SC:** LTE J max 5e-3. Pops: median 1.4e-3, max 7% at one height.
  - **SC with `--refine 2`:** J 9e-4, pops median 1.6e-4, max 6e-3, i.e.
    truncation from the extra y-crossing knots.
  - Promweaver in 3D: the vector form zeros at(0) only, so at(1) varies
    with y (as in dexrt_3d). Not changed.
  - **PRD Hybrid (static FAL-C, Longer):** converged pops agree with 2D to
    a median of 6e-5 and a max of 2.2e-3 (convergence level).
  - **MPI non-LTE:** 3D FAL-C, 8 iterations, Short (MC) and Longer, on CPU.
    3 ranks are bitwise identical to 1 rank.
  - **Pending:**
    - the dexrt_3d comparison (no 3D cube on this machine; `dex3d_1x.nc`
      is missing);
    - PRD on a 3D shear.
- **T7 notes.**
  - FAL-C 16×16×96, 24 dirs, 643 waves, RTX 4070: SC sweep 0.15 s/FS
    (2.5 G node-updates/s), Longer 0.06 s/FS (6.2 G/s). 2D reaches
    ~24 G/s on large rows.
  - Registers: the 3D SC sweep uses 116-119 (2D: capped at 64 by
    `LaunchBounds<512,2>`), the 3D Longer sweep 72-80.
  - CPU 3D SC: ~2 min/FS for 25k cells (slow).
- **T7 first pass (2026-10-25).** Benchmark: GivenFs 128×128×64, 24 dirs,
  24 waves, periodic x/y, RTX 4070 (scratchpad `t7/bench.py`). Sweep
  G node-updates/s:

  | step | SC MC | SC Linear | Longer |
  |---|---:|---:|---:|
  | baseline (119 regs) | 0.76 | 2.85 | 5.29 |
  | LaunchBounds<512,2> (64 regs) | 0.84 | 3.31 | |
  | χ/η interpolated from one fetch (`sc_interp_2d_pair`) | 1.31 | 4.84 | |
  | intensity ring (I, not I_nl + b S) | 1.40 | 4.94 | |
  | per-(dir, wave) pointers, i32 offsets, cheap periodic wrap | 1.99 | 6.74 | |
  | per-stencil axis offset tables (one wrap/clamp per axis) | 2.40 | 7.31 | 6.45 |

  - The ncu profile showed the kernel compute-bound, not memory-bound
    (DRAM/L1 ~10%). It was dominated by integer index arithmetic (55% of
    instructions at baseline), so shared memory is not the lever.
  - LaunchBounds at 80 or 128 registers is slower than at 64.
  - An interior fast path duplicated per material site was slower (code
    size and spills), so it is used only for the intensity stencil.
  - CPU (16 threads, `--bind-to none`, 64×64×32): MC 0.04, Linear 0.10,
    Longer 0.13 G/s. Most of the earlier CPU slowness was mpirun binding
    the rank to one core.
  - Next candidates:
    - Longer: store e0 or g in phase A rather than recomputing exp in
      phase B;
    - precomputed per-(dir, plane) crossing tables for SC (the pattern is
      the same for every interior node);
    - per-plane launches for small planes;
    - the CPU kernels.
- **Longer tiling (2026-10-25).**
  - ncu on the global-scratch kernel: DRAM 57%. The `I_seg` round trip
    was about half the traffic (6.1 GB written and 4.8 GB read per launch).
  - Tiled kernel (`sc_longer_sweep_3d_tiled`):
    - planes in tiles of up to `SC_LONGER_TILE_3D` (32)²;
    - each tile's lines, plus a halo of `lc_halo_x/y`, are marched with
      their segments in team shared memory;
    - the owner tile carries each line to the next plane;
    - fallback to the global kernel when the halo or segments don't fit.
  - Result: 3.3 GB written and 2.1 GB read (DRAM 32%), now compute-bound
    (ALU 49%, XU 12%).
  - Longer: 6.45 → 7.88 G node-updates/s.
  - Tried and dropped:
    - a single expm1 per segment (no change, kept as cleaner);
    - an interior-line fast path (slower, as for SC).
- **3D PRD in moving media (`tests/sc/sc_3d_prd.py`).**
  - Setup: H_6 PRD Hybrid on FAL-C at its native heights, periodic,
    Stepan 7.
  - **Longer:**
    - v_x(z) 20 km/s tanh shear, 3D vs 2D: median 6e-5, max 2.2e-3;
    - the same shear in v_y vs in v_x: median 5e-6;
    - genuinely 3D shear (v_x(y), v_y(x), 15 km/s): 298 its; 2 ranks vs 1:
      5e-7.
  - **SC:**
    - 3D vs 2D: median 2.7e-3, max 6.8% (the native-grid truncation);
    - v_y vs v_x: 5e-7;
    - 3D shear: 244 its; 2 ranks vs 1: 4e-7.
- **Plane splitting for occupancy (2026-10-25).**
  - `sc_3d_plane_chunks`: when the batch's (dir, wave) pairs are fewer than
    the device's resident teams (concurrency / team size), each plane is
    split over chunks teams per pair, one launch per plane. SC splits by
    node slices; the tiled longer kernel by tiles, with separate first-plane
    and leaving-line kernels. Otherwise there is one launch per task, as
    before.
  - Emulated memory-limited batch (256×256×32, wave_batch 2 = 24 pairs,
    4070): SC MC 1.11 → 2.28, Linear 3.08 → 5.45, Longer 2.73 → 5.43
    G node-updates/s. With the auto batch (7, 2 chunks): 2.43 / 6.64 /
    6.34.
  - SC results are bitwise unchanged, and longer results bitwise against
    the tiled kernel.
  - For 1024³ on 80 GB A100s (estimate): wave batch ~8 × 12 dirs = 96
    pairs against ~430 resident teams, so ~5 chunks.
- **Notes for later:**
  - SC crossing tables (per (dir, plane) crossing geometry: face type,
    offsets, fractions, step factors) for the next SC tuning pass;
  - a wavelength-group rank dimension (deferred; ≤16 4×A100 nodes
    expected);
  - distributed output for large cubes.
- **T5: in-plane rays (2026-10-26).**
  - `ScInPlane3d.cpp` solves each z plane as a 2D lattice of longer
    characteristics for both characteristics (cell means). The z drift is
    neglected.
  - Shared core: `ScLatticeRows.hpp` (`sc_lattice_rows<Blocked, Gather,
    TrackTau>`) is the 2D longer row algorithm, extracted. The 2D sweep now
    calls it (2D still bitwise, same speed).
  - In-plane row tables are built with `sc_longer_row_a`. The primary axis
    is a fixed axis if there is one, else the periodic axis with more cells.
    t may exceed 1 (multiple segments per row).
  - Periodic a:
    - the slope is snapped to m/N_a;
    - a march-only pass gives each line's lap map (τ, B);
    - the start values come from S(k) = e^-τ(k) S(k − m) + B(k), composed
      over N_b steps by pointer doubling (closes in log₂ N_b steps however
      thin the plane);
    - lines entering through a fixed b face end the chains (τ = ∞).
  - Tests (Healpix, 16 equatorial in-plane directions):
    - uniform medium vs the analytic J: 2e-7 (Short and Longer);
    - shift: 1e-3 (the lattice is anchored to the primary axis's first
      face);
    - all fixed/periodic combinations: J below the periodic case;
    - y-invariant 3D vs 2D: 1.8e-7, no y variation;
    - 2 ranks vs 1: 2e-7.
- **Horizontal classification changed (2D and 3D).** Laps of a periodic axis
  no longer make a ray horizontal: in-plane treatment gives an endless path,
  badly wrong in thin media. Found by the quadrature study on an 8×8 grid,
  where the μ ≈ 0.07 rays crossed more than the domain width per plane.
  - Horizontal now means: μz ≈ 0, or (longer) crossing a fixed axis's whole
    width within a step, or more than `SC_MAX_ROW_CROSSINGS` (1000) columns
    per step (a cost cap).
- **Quadrature convergence** (`scripts/sc/quadrature_convergence.py`):
  - Setup: plane-parallel stratified GivenFs slab, 16 wavelengths with slab
    τ from 1e-3 to 1e4, longer characteristics (exact per direction). The
    solver matches each quadrature applied to the exact solution to 1e-6.
  - Median relative J error by slab τ:

    | quadrature (N) | thin (<0.1) | mid | thick (>10) | max abs J err / S |
    |---|---:|---:|---:|---:|
    | Stepan 7 (24) | 0.54 | 2.8e-2 | 6.6e-4 | 0.08 |
    | Stepan 15 (88) | 0.36 | 7.7e-3 | 1.3e-5 | 0.03 |
    | Healpix nside 4 (192) | 1.6 | 2.0e-2 | 1.2e-4 | 0.08 |
    | Healpix nside 8 (768) | 0.66 | 9.6e-3 | 3.3e-5 | 0.04 |
    | Healpix nside 16 (3072) | 0.24 | 5.0e-3 | 6.6e-6 | 0.02 |

  - Healpix converges ∝ 1/N; Stepan is much better per direction.
  - In thin media, Healpix's μ = 0 ring (weight 1/(3 nside), I = S in a
    plane-parallel slab) dominates: J is overestimated by up to 24× at
    nside 4, where the true J ≪ S.
- **Default:** `sc.characteristics` is now LongerCharacteristics in 2D and
  3D. Tests that check SC pin "Short".
- **Healpix rotation and in-plane removal (2026-10-26).**
  - Healpix is rotated off μz = 0 by the deterministic rotation maximising
    min |μz| (Kronecker sequence over Euler angles; 2D: about y only, so the
    fold still applies).
  - Stratified-slab J errors are 2-14× lower than aligned. Sweep cost at
    nside 4 on 128×128×64: SC Linear +8%, MC +18%, Longer +22%.
  - The in-plane solver (`ScInPlane3d.cpp`) and the 2D horizontal-row
    solvers are removed. Rays with μz = 0, or crossing more than
    `SC_MAX_ROW_CROSSINGS` columns per step, are rejected with an error
    (2D and 3D). 2D AlongY (2.5D) remains. The shared lattice-row core
    stays for the 2D longer sweep; the in-plane version is in df39c4b if it
    is needed again.
- **3D longer kernel (2026-10-26).** The tiled shared-memory kernel lost
  badly on stretched grids. FAL-C 64×64×96 has 55 km steps against dx =
  20 km: about 14 segments, halo 10, the 48 KB Kokkos team-scratch cap (no
  opt-in in this Kokkos) forced 16 tiles, and it marched 2.6 lines per cell
  (4.27 s against 2.01 s for the global kernel).
  - Tried:
    - a per-plane halo: bitwise identical, 3.10 s FAL-C, 7.80 G/s uniform;
    - an index-synchronous accumulate (slower everywhere: barrier per
      segment index, lock-step idling).
  - Now: **one atomic pass per plane** (`sc_longer_sweep_3d_atomic`).
    - Lines march all segments and add I len g / path into I_nl with
      `dex_team_add` (atomic on GPU, a plain add on host).
    - Cells form their own b, q from χ and the plane's segment lengths (the
      path per cell is the same for every cell on a plane).
    - No segment store, no gather, no blocking.
    - Results: FAL-C 1.87 s, uniform 7.56 G/s, low-occupancy split (wave
      batch 2) 6.28 G/s (tiled 5.43), CPU 0.19 G/s (was 0.13).
    - Replaces both the tiled and the global kernels (−287 source lines).
    - GPU results now vary run to run at ~1e-7 (atomic summation order).
  - FAL-C non-LTE: Longer 1.82 s/it vs SC 1.92 (was 2.56 with the tiled
    kernel). PRD vs CRD: ~2.2× in 3D, as in 2D.
