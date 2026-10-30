# PRD in dexrt_sc: hybrid approximation (plan for critique, 2026-10-25)

## Goal
Add partial redistribution (PRD) for lines flagged `PRD-Voigt` in CRTAF
(the H_6 Lyα/Lyβ, Ca II H&K, later Mg II h&k). Use the hybrid approximation
of Leenaarts et al. (2012b), as Sukhorukov & Leenaarts (2016, "SL16") and
Lightweaver do: an angle-averaged R_II in the comoving frame, with an
isotropic comoving-frame J.

The method has to hold up in 3D, where SC will end up. So we store no
per-(cell, direction, wavelength) interpolation coefficients, as Lightweaver
does with `JCoeffs` and `hPrdCoeffs`. We also store no per-cell G_II
arrays, as Lightweaver does with `gII(k, la)`. And we don't use SL16's
equidistant real/virtual fine grid with its five bookkeeping arrays.

v1 scope:
- Resonance PRD only, with no cross-redistribution (XRD). This matches
  Lightweaver's `GII` (waveratio = 1), so the two can be compared.
- The coherence fraction is in the impact limit:
  γ = P_j / (P_j + Q_E).
  This is wrong for the Lyman lines, which need a wavelength-dependent
  γ(Δλ). The kernel therefore evaluates γ per (cell, emission wavelength)
  behind one function (see 4), so that a later Lyman γ(q) changes only that
  function.

## The equations (SL16 Eqs. 5–18, in our notation)
For a PRD line i→j, with ρ the profile ratio ψ/φ:
- Lab-frame emission profile: ψ(n, λ) = ρ(n, λ) φ(n, λ).
- η_line = (hν/4π) n_j A_ji ρ φ.
- χ_line = (hν/4π) φ (n_i B_ij − ρ n_j B_ji).
- The rates use J̄_φ (R_ij), and J̄_ψ = J̄ weighted by ρφ (R_ji).

In the hybrid approximation ρ comes from the comoving frame:
- **Forward:** J*(λ*) = Σ_μ w_μ I_μ(λ* shifted to the lab frame), on the
  comoving knots.
- **Redistribution, per cell:**
  ρ*(λ*_e) = 1 + γ(λ*_e) · n_i B_ij / (n_j P_j) ·
  [ ∫ J*(λ') g_II(λ', λ*_e) dλ' / ∫ g_II dλ' − J̄_φ ].
  - g_II = R_II/φ is Gouttebroze's approximation, in local Doppler units q.
  - The integral is normalised as Lightweaver does (HM14 §15.4).
  - In the impact limit, γ · n_i B_ij / (n_j P_j) =
    n_i B_ij / (n_j (P_j + Q_E)), which is Lightweaver's `gammaPrefactor`.
    The general form is kept so that γ can depend on q.
- **Backward:** ρ(n, λ) = ρ*(λ*), where λ* is the comoving wavelength
  already formed for the Voigt evaluation, v = (λ − λ0 + v_los λ0/c)/Δλ_D in
  `ScEmisOpac::profile`. The PRD lookup uses the same shift as φ, by
  construction.

J̄_φ is computed from J* with the static comoving profile:
J̄_φ = ∫ J* φ* dλ*. In the hybrid picture this is identical to R_ij/B_ij,
apart from the discretisation. It keeps ρ self-consistent: ρ = 1 exactly
when J* is flat across the profile. Lightweaver uses the accumulated R_ij
instead, so (P4) reports the difference.

## Why this is simpler than SL16 / Lightweaver
Each "forward transform" in SL16 is a scatter from one lab frequency at a
time, because Multi3D only has one frequency of I in memory. That ordering
is what forces their address/forward_j arrays, or the equidistant grids.

We don't have that constraint. A DAG task holds I for a whole wavelength
batch and all directions in its batch, as `ScWorkspace` (I_nl, b, q, S).
So we write the forward transform as a **gather on the comoving knots**:
- For each (cell, direction), the shift u = v·n/c is constant.
- The comoving PRD knots whose lab positions λ*_j(1 − u) fall inside this
  batch's lab range form a contiguous run of j. Two binary searches on the
  (≤ a few hundred) PRD knots find it.
- Walk j upwards with a hunting pointer i into the batch's lab knots, and
  add w_μ [(1 − t) I_i + t I_{i+1}].

Linear interpolation is a sum of hat functions:
I(λ) = Σ_i h_i(λ) I_i, where h_i is the triangle on (λ_{i−1}, λ_{i+1}).
Take a comoving knot j whose lab position falls between lab knots i (the
last knot of batch b) and i + 1 (the first knot of batch b + 1):
- Batch b adds w_μ (1 − t) I_i.
- Batch b + 1 adds w_μ t I_{i+1}.
- t needs only the global grid, never I from the other batch.

So each batch covers the comoving knots inside the hat supports of its own
knots, (λ_{first−1}, λ_{last+1}). There is no rolling or carried state
between batches, the order of batches doesn't matter, and nothing is stored.

This works only because linear interpolation has a one-knot support. A
higher-order or limited interpolant would need an I halo across batch
edges, so the forward transform stays linear.

Multi3D needs its equidistant grids only because of its loop order. It
sweeps one frequency at a time, so each (point, dir, frequency) has to
search afresh, and a correlated search would need stored per-(point, dir)
state (SL16 footnote 3). Here the two searches are shared across the batch.

At the grid ends we use constant extrapolation (the end hat extends to ±∞),
as SL16 Fig. A.1. The backward transform is a plain interpolation of ρ* at
λ*. It is one search on the line's knots, done once per
(cell, dir, wave, PRD line), next to a Voigt LUT evaluation that we already
pay for.

Threads own cells: the Gamma/J kernels already accumulate J non-atomically
per run piece. J* is filled the same way, so it needs no atomics.

## G_II: compute, don't store
Gouttebroze's g_II(a, q_e, q_a) splits as

  cf(a, q_e) · core(q_e, q_a) + (1 − cf(a, q_e)) · wing(q_e, q_a)

where core and wing do not depend on a, and cf ≠ 0, 1 only for
2 ≤ |q_e| ≤ 4. That makes it a universal function in Doppler units: no
per-line or per-cell table is needed, should we ever tabulate it.

A direct evaluation costs 1–2 exp, 1 sqrt and about 15 flops, which is
about the cost of a bilinear lookup. v1 evaluates directly, and we
tabulate core and wing (2D, in q_e and q_a − q_e) only if profiling shows
the redistribution kernel is compute-bound.

The Voigt profile is already a LUT (`VoigtProfile<..., USE_LUT>`), so
nothing new is needed there. PRD adds only the ρ lookup.

## Components
1. **Atom data (`CrtafParser.hpp`, `Types.hpp`).**
   - Parse the CRTAF `elastic` flag on each broadening term. It is already
     in H_6.yaml and CaII.yaml, and the parser currently ignores it.
   - Q_E(cell) = Σ of the elastic terms. Natural broadening is not elastic.
     This mirrors Lightweaver's `Qelast`.
   - Device-side: a list of PRD lines (kr), and the union of their global
     wavelength knots `prd_la` → index into the J* array.
2. **State (`ScPrd` struct, new `ScPrd.hpp/.cpp`).**
   - `J_rest [prd_la, ks]`.
   - `rho [prd line, line wave, ks]`. This is ρ* on the line's own knots,
     `blue_idx..red_idx`, initialised to 1 (CRD).
   - `P_j [prd upper level, ks]`: the total radiative out-rate of each PRD
     upper level, accumulated in the Gamma kernel. That is 4 floats per cell
     for H_6 + Ca II, not the full `RateDiagnostics` GammaMat (72).
   - PRD knots: Lyα 99, Lyβ 71, Ca II H 49, Ca II K 49, for 268 ρ* plus a
     268 J* union. That is about 2.1 KB/cell in f32.
   - Memory, Snow KHI 2D: 194k cells × 2.1 KB ≈ 0.4 GB.
   - f16 storage of J* and ρ* follows the `SC_HALF_RADIATION` option (see
     "Half-precision radiation storage" below).
   - For a 1024³ estimate see "Memory at scale" below.
3. **Material and Gamma with ρ (`ScEmisOpac`, `ScGammaAccumulate`).**
   - `ScEmisOpac::evaluate` multiplies the PRD line's emission by ρ, and its
     stimulated-emission opacity by ρ.
     - χ* = n_i B_ij − n_j B_ji currently arrives pre-combined from
       `compute_core_and_voigt`.
     - Split out the stimulated part as η* × (B_ji/A_ji), a per-line
       constant. That needs no extra storage.
     - Check against `compute_core_and_voigt` in a unit test.
   - Line U/V in the Gamma accumulation: U_ji and V_ji get × ρ, and V_ij is
     unchanged.
   - A non-PRD line takes no extra cost: a per-line flag skips the lookup.
4. **Redistribution kernel (`sc_prd_redistribute`).**
   - One thread per (cell, PRD line). On CPU it runs over cell groups,
     looping over the lines.
   - For each emission knot j of the line:
     - q_e = (λ_j − λ0)/Δλ_D(cell);
     - take the integration range from Lightweaver's `scattering_int_range`;
     - walk a uniform fine grid of Δq = 0.15, with ≤ ~95 points;
     - interpolate J* there, hunting on the line's knots in q;
     - accumulate g_II·w and J·g_II·w;
     - set ρ*[j] = 1 + γ(cell, q_e) · n_i B_ij / (n_j P_j) · (S/N − J̄_φ).
   - P_j = Σ_k (R_jk + C_jk), from the rates and the existing collisional
     matrix C. γ = `sc_prd_coherence(line, cell, q_e)`, which is the impact
     limit in v1.
   - Cost: N_line_wave × ~80 g_II evaluations per (cell, line). For Snow
     KHI with 2 PRD lines that is about 194k × 2 × 150 × 80 ≈ 5 G, roughly
     tens of ms on the 4070. That is cheap next to a formal solution.
5. **Forward transform (`sc_accumulate_J_rest`).**
   - A separate kernel after Gamma/J in the DAG task. It runs only for
     batches that overlap the PRD region, and it re-reads I = I_nl + b S
     from the workspace.
   - It is kept out of the existing Gamma kernel so that kernel stays
     untouched.
   - The archived rest-frame back-map (`sc-rest-frame` tag) was GPU-divergent
     because its work per thread varied with velocity. Here the work per
     (cell, dir) is a contiguous walk of ≤ (batch knots + run length).
     Measure it. Fold it into the Gamma kernel only if the extra read
     matters.
   - Reuse `sc_find_bin` and `sc_find_bin_near` from
     `sc-rest-frame:source/ScSpectralRemap.hpp`. Nothing else from that
     code is needed.
6. **Iteration (`ScMain.cpp`).**
   - Per non-LTE iteration:
     1. Full formal solution: Gamma, J, J*, rates.
     2. stat_eq / NR / conservation / Ng, as now.
     3. N_sub PRD sub-iterations, each with fixed pops:
        (a) redistribute, giving ρ* from J*;
        (b) a formal solution restricted to the "PRD-reach" wavelength
            batches, accumulating J* and the PRD lines' rates only, with no
            Gamma.
        Stop the sub-iterations when max |Δρ*| < tol.
   - The PRD reach is the union lab range [λ_lo(1 − u_max), λ_hi(1 + u_max)]
     plus one knot on each side. u_max is the global max |v| (allreduce),
     so every rank has the same task layout, which `validate_task_layout`
     requires.
     It plays the role of Lightweaver's `hPrdIdxs`. The DAG `run` gains a
     wavelength-batch subset, and batches stay as chosen now.
   - SL16: one sub-iteration always diverges, Mg II needs 2 and H I needs
     3. They are optional YAML overrides in the `sc:` block:
     - `prd_subiterations` (default 3);
     - `prd_tolerance` on max |Δρ*| (default 1e-3);
     - `prd_start_iter`, the first iteration with ρ ≠ 1 (default 0). SL16
       recommends a CRD or zero-radiation start so that Ng behaves.
   - Also add a run-time `sc.prd: Off | Hybrid` (default Off; user,
     2026-10-25). It allows CRD comparisons and leaves room for a later
     angle-dependent mode.
7. **MPI / DAG.**
   - Everything new is per cell and rank-local. J* and ρ* are owned like J,
     and P_j and Q_E are local.
   - The ghost-input snapshot must carry ρ* for ghost rows, because the
     material in halo cells uses it. ρ* is refreshed at the same epochs as
     the pops ghosts.
   - Sub-iteration formal solutions reuse the fused-task pipeline over the
     batch subset.
8. **Output and restart.** Write ρ* per PRD line (sparse or dense as
   configured) and J* behind an output flag. Put ρ* in the snapshot so that
   restarts are exact.
9. **dexrt_py.** Add the schema for `sc.prd`, reader support for the `rho`
   variables, and a CRTAF Mg II atom with h&k flagged PRD.

## Classical J and half-precision radiation storage
The full-spectrum J, `[num_wave, ks]` (3.6 KB/cell for H_6 + Ca II), is the
largest single SC array, and SC barely uses it:
- output, if `output.J`;
- background scattering (`update_background` copies J into
  `background.J_prev`, a second full-spectrum array);
- restart snapshots, for the background J_prev.

1. **Allocate J only when needed.** That is when `output.J` is set, or
   background scattering is enabled. Otherwise J is a 0-size array, and the
   accumulation is compiled out by a runtime flag already passed to the
   Gamma kernel (`accumulate_J`). The memory model (`sc_choose_wave_batch`)
   counts what is allocated.
2. **`SC_HALF_RADIATION`** is a constexpr bool in `ScConfig.hpp`, default
   false. It stores J, `background.J_prev`, J* and ρ* as 16-bit values.
   - **Storage.** `ScRadArray` (`ScHalf.hpp`): a `u16` array with portable
     bit-level conversions (round to nearest even), else `fp_t`.
     - Kokkos `half_t` is a float on host backends, so it saves nothing on
       CPU.
     - The conversions are used only inside SC accessors: `ScRadStore`
       with `load(la, ks)` and `add(la, ks, v)`. Nothing else sees the
       storage type.
   - **Range (as built).** J, J* and ρ are all non-negative, so the format
     is unsigned: a 6-bit exponent and binary16's 10-bit mantissa (2⁻¹¹
     relative). Normal values span [2⁻³⁹, 2²⁴) and subnormals reach 2⁻⁴⁹.
     That covers intensities with no per-wavelength scales and no
     saturation bookkeeping, and ρ is stored directly. Negative or NaN
     inputs store 0, and values beyond the range saturate. This replaces
     the scaled IEEE f16 design.
   - **Precision.** Accumulate in f32 inside a task, and write once per
     (wavelength or knot, cell, task):
     - J: the Gamma kernel already sums the batch's directions in f32
       (`J_entry`).
     - J*: loop j outer and directions inner.

     That is one f16 read-modify-write per direction batch, a few per
     formal solution.
   - **I/O.** The writer converts to f32 with the scale, so files are
     unchanged. Snapshots store f32.
   - **Validation.** With the option off, output is bitwise unchanged. With
     it on, compare pops and J against f32 on Snow KHI and the P3 PRD cases
     at the non-LTE tolerance.
3. **Order.** (1) and the J half of (2) don't depend on PRD, so they go
   first, as stage P0.

## Memory at scale (1024³ Bifrost, H_6 + Ca II, f32, all cells active)
These are estimates. 3D SC isn't built yet, so the workspace figure is the
2D layout extrapolated.

| Per cell | B |
|---|---|
| atmos (T, ne, vturb, nH, nH0, v) | 40 |
| pops + n* (12 levels) | 96 |
| Gamma (2 × 6²) | 288 |
| Ng history (~6 × pops) | 288 |
| ScEmisOpac line params (15 lines × 4) | 240 |
| ScEmisOpac continua (2 × wave batch of 49) | 390 |
| **Base, non-LTE CRD** | **~1.35 KB** |
| PRD J* + ρ* + P_j | ~2.1 KB (~1.1 KB with `SC_HALF_RADIATION`) |
| **Persistent total** | **~3.5 KB → 3.7 TB** |
| SC workspace (χ, S, I_nl, Δτ per dir × wave in batch) | 16 B × batch |

- **The full-spectrum J is not counted.** 895 waves would be 3.6 KB/cell,
  1.8 KB in half precision. It is allocated only for output or background
  scattering (see above).
- **The workspace is elastic.** The wave-batch chooser sizes it to the
  memory left over. A batch of 16 dirs × 24 waves costs about 6 KB/cell;
  the pipeline floor is about 1 KB/cell.
- **Example layout.** 32 nodes × 4 H100 (80 GB), so 128 GPUs with 8.4M
  cells each, e.g. 256 × 256 × 128 blocks.
  - Persistent state is about 29 GB/GPU, leaving about 45 GB of workspace.
  - That is comfortable. 16 nodes (about 58 GB of state per GPU) is the
    tight lower end.
- **Easy reductions:**
  - Drop the inactive corona cells (sparse active map) and the deep
    convection zone below a Thermalised boundary. Together these are
    commonly 30–50% of a Bifrost box.
  - ρ* in f16.
  - SL16-style truncated PRD wings (their Fig. 3).
- **Comparison with SL16.** They avoided storing I, which was 1–4 GiB per
  32³ subdomain. We never hold I beyond one batch, and PRD costs us
  J* + ρ*, which is the same as their J* + ρ* storage.

## Stages and validation
- **P0, J allocation and half storage.**
  - J is allocated only when it's needed.
  - `SC_HALF_RADIATION` for J and J_prev.
  - Unit tests: f16 conversion (round trip, rounding, saturation,
    subnormals), and scale selection.
  - Off: bitwise unchanged. On: compare with f32 on Snow KHI.
- **P1, kernels (unit, `test_sc.cpp`).**
  - g_II against a straight port of Lightweaver's `GII` (f64, exact).
  - Redistribution of a flat J*: ρ* = 1 to rounding, from the normalisation
    and the J̄_φ term.
  - Forward transform:
    - at v = 0 it is the identity, J* = J on the PRD knots;
    - an isotropic, spectrally linear I gives J* exactly;
    - the result is independent of the batch split (bitwise for one
      direction; rounding only otherwise);
    - constant extrapolation at the ends.
  - Backward transform: ρ = ρ* at v = 0.
  - χ split: η*·B_ji/A_ji + (χ* + that) reproduces the
    `compute_core_and_voigt` terms.
- **P2, CRD regression.**
  - With PRD forced Off, Snow KHI output is bitwise unchanged.
  - With Hybrid and γ forced to 0, ρ ≡ 1 and the result equals CRD to
    rounding.
- **P3, the paper's 1D tests via a tiled periodic FAL-C**
  (`lw_compare_native_z.py` setup, H_6 with Lyα/β PRD, then Ca II).
  - Velocities: v_z = 0, ±10 km/s constant, ±10 gradients, toy shocks.
  - Compare emergent profiles and pops with Lightweaver 1D hybrid PRD.
  - The constant-v cases must match the v = 0 case shifted (Galilean). This
    is the cleanest check of the forward/backward pair.
- **P4, convergence.**
  - Sub-iterations 1–4, LTE vs zero-radiation start, and Ng on or off.
    Reproduce SL16 Table 1 qualitatively.
  - Report J̄_φ from J* against R_ij/B_ij.
- **P5, MPI/DAG.** 1, 2 and 4 ranks agree to rounding (J* accumulation order
  changes across rank counts, as for the rest-frame work).
- **P6, Snow KHI performance.** Timers for the redistribution, forward
  transform and ρ lookup, and the sub-iteration formal solution cost
  against a full formal solution. Tabulate g_II here if it is hot.
- **Later.**
  - Wavelength-dependent γ for the Lyman lines.
  - XRD (Lyβ → Hα, Ca II H&K → IR triplet; Uitenbroek 1989), which sums
    over the subordinate lines k→j sharing the upper level.
  - Angle-dependent PRD as a reference, 1D only.

## Decisions (2026-10-25, user)
1. Sub-iteration formal solutions cover only the PRD-reach batches, as
   Lightweaver's `hPrdIdxs`.
2. J̄_φ is computed from J* (local, self-consistent). P4 compares it with
   R_ij/B_ij.
3. γ is the impact limit, as Lightweaver, for now. The Lyman γ(Δλ) will be
   revisited behind `sc_prd_coherence`.
4. Sub-iterations, tolerance and start iteration are optional YAML
   overrides (`sc.prd_subiterations`, `sc.prd_tolerance`,
   `sc.prd_start_iter`).
5. The comoving grid is the lines' own knots. Thinning is an atom-grid
   question (P3), not a sparser J* grid.
6. A compile-time f16 storage option, `SC_HALF_RADIATION`, covers the
   classical J (and the background J_prev) as well as J* and ρ*. The
   classical J is also allocated only when it is used.

## Status (2026-10-25)
- **Done.**
  - J is allocated only for output or the background.
  - `elastic` is parsed from CRTAF.
  - `ScPrd` (`ScPrd.hpp/.cpp`):
    - the line table;
    - ρ* lookup (material, and Gamma U_ji/V_ji);
    - the I → J* gather (one store per (knot, task));
    - redistribution (Gouttebroze g_II, J̄_φ from J*, impact-limit γ);
    - collisional and radiative P_j (radiative from `rate_diag`, allocated
      internally);
    - the sub-iteration loop over the reach range only (tasks clipped);
    - `prd_rho` and `prd_lines` output;
    - ρ* in the ghost inputs (MPI).
  - `ScRadArray` storage for J, J_prev, J* and ρ*, with
    `SC_HALF_RADIATION` (default off).
- **Tests.**
  - Unit tests for g_II (against Lightweaver), the ρ lookup, and the
    16-bit format.
  - PRD on 1/2/3 ranks agrees to 5e-7 (`tests/sc/sc_prd_mpi_test.py`).
- **Resolved: CRD/PRD offsets and the single-sub-iteration divergence.**
  - Both came from the opacity-weighted PL step (f9e97e9). It is now the
    `PiecewiseLinearEmissivity` formal solver (still the default).
  - `PiecewiseLinear` is the standard step, as in Lightweaver.
  - With `PiecewiseLinear` on FAL-C H_6 (Lightweaver with linear
    collision-rate interpolation), level 1 median vs Lightweaver:
    - CRD: 1.2e-3 (max 7%, at 729 km);
    - PRD: 1.1e-3 (max 10%).
  - `prd_tolerance` 1e-2 converges in 245 iterations (Lightweaver 248).
  - Lyα ρ*: median |Δρ| 5e-6, 99% 2.6e-2. The worst points are the
    outermost far-wing knots near 700 km.
- **Fixed: forward transform at the line edges.**
  - J* interpolated the lab intensity from the global grid. At a moving
    cell's outermost comoving knot, that picked up the next lab knot outside
    the line, which has no line opacity: J there is 7000x larger at
    122.41 nm next to Lyα's red end.
  - The result was ρ ~ 1e8 for v_z = 10 km/s, constant or sheared.
  - J* now uses only the line's own lab knots, with constant extrapolation,
    as ρ_lab does. That also makes the sub-iteration wavelengths just the
    lines' own knots (no v_max).
  - Lightweaver's hybrid coefficients mix in the same knot (its ρ reaches
    473 there in the shear case). This is a candidate fix for Lightweaver.
- **Moving media (FAL-C H_6, v_z = 10 km/s; `PiecewiseLinear`).**
  - Constant v: 245 iterations, ρ ≤ 2.81 (static: 2.87).
  - Shear 10 sin(2πx/L) km/s: 245 iterations with 3 sub-iterations per
    iteration.
- **One sub-iteration per iteration leaves a slow mode.**
  - The mode sits at ~1e-3 on H II (T ≈ 6900 K), driven by Lyα.
  - Lightweaver has it too: run to 1e-6 it stalls at 7.6e-4 after 699
    iterations.
  - It is not caused by stale P_j (refreshing the PRD lines' rates didn't
    help, and was reverted), nor by J̄ from J* vs R_ij/B_ij.
  - Fix: `SC_PRD_MIN_SUBITERATIONS = 2` (constexpr). The tolerance only ends
    sub-iterations after the second.
  - Static and shear cases, at tolerances 1e-2 and 1e-3, now converge to
    the f32 floor (~2e-6, as CRD).
- **Fixed during validation.**
  - The PRD-only solves skipped `update_background`, so owned and ghost
    background populations disagreed (MPI).
  - `ScPrd::enabled()` was not device-callable (GPU Gamma garbage).
- **Not done.**
  - Restart of ρ*.
  - CUDA build check.
  - `SC_HALF_RADIATION` validation runs.
  - Shear/velocity cases.
  - Performance.
