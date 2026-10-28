#if !defined(DEXRT_STATE_SETUP_HPP)
#define DEXRT_STATE_SETUP_HPP

#include "Types.hpp"
#include "State.hpp"
#include "DexrtConfig.hpp"
#include <string>

/// Solver-independent State setup and I/O shared between the 2D executables.

void allocate_J(State* state);
void allocate_rad_loss(State* state);
/// Rows [z_start, z_end) of the atmosphere (e.g. one rank's slab).
struct AtmosZWindow {
    i32 z_start;
    i32 z_end;
};

/// Loads atoms and atmosphere (sparsified with max_mip_level from config), and
/// allocates pops/Gamma/wphi/bcs. Returns the C0 ray configuration for RC. With
/// a window, only those rows of the (dense) atmosphere are loaded.
CascadeRays init_atmos_atoms(State* st, const DexrtConfig& config, const AtmosZWindow* window = nullptr);
/// Loads eta/chi for DexrtMode::GivenFs.
CascadeRays init_given_emis_opac(State* st, const DexrtConfig& config);

/// Loads the populations, ne, and nh_tot from restart_path. Returns the current
/// iteration number.
int handle_restart(State* st, const std::string& restart_path);
void save_snapshot(const State& state, int num_iter);
void load_initial_pops(State* st, const std::string& initial_pops_path);

void copy_J_plane_to_host(const State& state, int la_start, int la_end);
void copy_rad_loss_plane_to_host(const State& state, int la_start, int la_end);

#else
#endif
