# Future work: index `Gamma` by gamma atom

## What this is

`state.Gamma` holds one matrix per atom, allocated over **every** atom in
`adata`, and is indexed by position in `adata` everywhere. `Detailed` atoms
therefore get a matrix allocated, get radiative rates accumulated into it by the
formal solver, and get their populations overwritten by `stat_eq` /
`time_dep_update` — none of which should happen for an atom whose populations are
prescribed.

The better data model is to size and index `Gamma` (and `state.rate_diag.*`) by
position in `atoms_with_gamma`, using `atoms_with_gamma_mapping[ig]` for every
`adata` / `adata_host` lookup. `Detailed` atoms then simply have no matrix.

This was deliberately not done when the indexing mismatch in
`compute_collisions_to_gamma` was fixed (that fix made the outlier conform to the
adata convention instead, as a one-liner). It is a behaviour change across eight
sites in both 2D and 3D, so it wants doing on its own.

## Sites to change

Each of these currently uses an `adata` index into `Gamma`:

| site | notes |
|---|---|
| `main.cpp`, `main_3d.cpp`, `main_rad_eq.cpp`, `DexInterface.cpp::allocate_cell_count_based_terms` | allocation; loop over `atoms_with_gamma`, take `num_level` through the mapping |
| `Populations.cpp::stat_eq_impl` (both the MAGMA and kokkos-kernels variants) | `abundance(ia)`, `level_start(ia)`, `num_level(ia)` all need mapping |
| `TimeDepPopulations.cpp::time_dep_impl` | as above |
| `StatEqOptions::only_atom` / `KineticEqOptions::only_atom` | callers (`InitialPops`) pass adata indices; pick and document one convention |
| `InitialPops.cpp::set_zero_radiation_pops` | takes an atom index, reads `Gamma[atom]` |
| `ChargeConservation.cpp` | `Gamma[0]` assumes H is adata atom 0 (`have_h_model` is `models[0].Z == 1`); needs H's *gamma* index, and a guard for H not being active |
| `DynamicFormalSolution.cpp` (`_nonatomic` and `_atomic`), `DynamicFormalSolution3d.cpp` | loop bound becomes `atoms_with_gamma.size()`, all `adata.*(ia)` lookups go through the mapping |
| `RateDiagnostics.hpp` (`allocate`, `write_rate_diagnostics`) | same as the formal solver |

Already index-agnostic, needing no change: the `Gamma.size()`-based zeroing loops,
`WavelengthDistributor::reduce_Gamma`, `reduce_rate_diagnostics`.

## Output impact

`radiative_rates_<ia>` / `collisional_rates_<ia>` / `cont_energy_*_<ia>` would be
numbered by gamma index, which no longer lines up with the `num_level` and
`level_start` global attributes (those are per adata atom). Write the mapping out
as a `gamma_atom_index` global attribute so consumers can map back, and update
`instrumented_rerun_dex_handoff.md`, which currently documents `<ia>` as the index
into the config's atom list.

## Verification

The change is invisible to any config using only `Active` / `Golding` atoms, so
existing runs should be bit-identical — `DexRT/TestJob/check_rate_diag.py` already
does a pairwise bit-identity comparison and can be pointed at a before/after pair.

The behaviour change needs its own test: a config with a `Detailed` atom, checking
that its populations are *not* modified by the solve, and that the active atoms'
rates are unaffected by its presence. The detailed-balance identity
`C_ij n_i* == C_ji n_j*` in `check_rate_diag.py` is a good probe for the atom
matrices having stayed correctly paired.
