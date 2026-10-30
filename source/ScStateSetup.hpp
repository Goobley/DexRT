#if !defined(DEXRT_SC_STATE_SETUP_HPP)
#define DEXRT_SC_STATE_SETUP_HPP

#include "Types.hpp"
#include "ScState.hpp"
#include "DexrtConfig.hpp"
#include <string>

/// dexrt_sc's state setup and I/O.

/// Loads atoms and this rank's rows of the atmosphere (state.slab; read by
/// rank 0), dense with void cells zeroed, and allocates pops/precon_R/wphi/bcs.
void init_atmos_atoms(ScState* st, const DexrtConfig& config);
/// Loads this rank's rows of eta/chi for DexrtMode::GivenFs.
void init_given_emis_opac(ScState* st, const DexrtConfig& config);
/// J is only kept for output and the background scattering.
void allocate_J(ScState* state);
/// rad_loss (config.rad_loss), over the owned rows.
void allocate_rad_loss(ScState* state);

/// Loads the populations, ne, nh_tot (and temperature, pressure, PRD rho, J
/// for the background, if present) from a dense snapshot at restart_path.
/// Returns the current iteration number.
int handle_restart(ScState* st, const std::string& restart_path);
void save_snapshot(const ScState& state, int num_iter);
/// Dense populations [level, z, (y,) x].
void load_initial_pops(ScState* st, const std::string& initial_pops_path);

#else
#endif
