#if !defined(DEXRT_STATE_SETUP_HPP)
#define DEXRT_STATE_SETUP_HPP

#include "Types.hpp"
#include "State.hpp"
#include "DexrtConfig.hpp"
#include "Populations.hpp"
#include <string>

/// RC State setup and I/O, and the setup shared with ScState.

/// Loads the atoms: adata, and the atoms with a Gamma and their mapping.
template <typename StateT>
void init_atomic_data(StateT* st, const DexrtConfig& config);
/// Allocates pops (and precon_R, rate diagnostics, wphi if non-LTE) over the
/// loaded atmosphere, and loads the boundary and background.
template <typename StateT>
void init_atom_storage(StateT* st, const DexrtConfig& config);

void allocate_J(State* state);
void allocate_rad_loss(State* state);
/// Loads atoms and atmosphere (sparsified with max_mip_level from config), and
/// allocates pops/precon_R/wphi/bcs. Returns the C0 ray configuration.
CascadeRays init_atmos_atoms(State* st, const DexrtConfig& config);
/// Loads eta/chi for DexrtMode::GivenFs.
CascadeRays init_given_emis_opac(State* st, const DexrtConfig& config);

/// Loads the populations, ne, and nh_tot from restart_path. Returns the current
/// iteration number.
int handle_restart(State* st, const std::string& restart_path);
void save_snapshot(const State& state, int num_iter);
void load_initial_pops(State* st, const std::string& initial_pops_path);

void copy_J_plane_to_host(const State& state, int la_start, int la_end);
void zero_rad_loss(const State& state);
void finalise_rad_loss_batch(const State& state, int la_start, int la_end);
void copy_rad_loss_to_host(const State& state);

#else
#endif
