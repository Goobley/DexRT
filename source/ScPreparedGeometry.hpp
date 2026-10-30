#ifndef DEXRT_SC_PREPARED_GEOMETRY_HPP
#define DEXRT_SC_PREPARED_GEOMETRY_HPP
#include "ScConfig.hpp"
#include "ScAngularQuadrature.hpp"
#include "ScDecomposition.hpp"
#include "ScLongerGeometry.hpp"
#include <vector>
#include "ScGeometry.hpp"

/// A batch of directions sharing a ray mode (and, for Sweep, the sweep
/// direction in z), solved together.
struct ScDirBatch {
    ScLongerGeometry longer;
    i32 num_dirs;
    ScRayMode mode;
    bool flip_z; /// Sweep only: sweeping downwards in z
    std::vector<i32> dir_idx_host;
    std::vector<ScFrame> frames_host;
    yakl::Array<i32, 1, yakl::memDevice> dir_idx; /// index into ScDirectionSet
    yakl::Array<ScFrame, 1, yakl::memDevice> frames;
    /// Sweep only: columns crossed and path length stepping into canonical row
    /// p [dir, p] (per row for a non-uniform z grid)
    Fp2d row_t;
    Fp2d row_ty; /// 3D: as row_t, along y
    Fp2d row_ds;
    /// Sweep: the path through canonical row p's cell (its height, between the
    /// midpoint faces, over |mu_z|) [dir, p], for rad_loss.
    Fp2d row_cell_ds;
};

/// 2D/3D: group directions by ray mode and z sweep direction into batches of at
/// most SC_DIR_BATCH. row_steps: sc_row_steps (empty: uniform dz).
/// max_step: the largest row step over the whole grid (so every slab makes the
/// same ray-mode decisions; 0: the largest of row_steps).
std::vector<ScDirBatch> sc_make_dir_batches(
    const ScGrid& grid,
    const ScDirectionSet& dirs,
    bool periodic_x,
    const std::vector<f64>& row_steps = {},
    f64 max_step = 0.0);

/// Largest direction batch (sizes the shared job buffers).
i32 sc_max_batch_dirs(const std::vector<ScDirBatch>& batches);

/// Flat extents of the independent Short boundary arrays, per direction/wavelength.
struct ScShortInflowSizes {
    i64 bottom = 0, edge = 0, ghost = 0, side = 0;
    i64 xface = 0, yface = 0, ghost_x = 0, ghost_y = 0;
    i64 total() const { return bottom + edge + ghost + side + xface + yface + ghost_x + ghost_y; }
};
struct ScLongerInflowSizes {
    i64 bottom = 0, side_x = 0, side_y = 0;
    i64 total() const { return bottom + side_x + side_y; }
};

/// Derived boundary extents of the numerical kernels. DDA traversal itself
/// needs no maximum-length chain allocation.
struct ScStorageRequirements {
    i32 side_x = 1, side_y = 1;
    ScShortInflowSizes short_inflow;
    ScLongerInflowSizes longer_inflow;
};

/// Immutable across solve epochs. Geometry owns all direction/path setup;
/// callers choose a wavelength width from its requirements, not vice versa.
struct ScPreparedGeometry {
    ScGrid grid;
    ScSlab slab;
    ScDirectionSet dirs;
    std::vector<ScDirBatch> batches;
    ScStorageRequirements storage;
    ScLongerLayout longer_layout;
    ScCharacteristics method;
    bool periodic_x, periodic_y;
    i64 buffer_values_per_pair() const;
    ScPreparedGeometry(
        const ScGrid& grid,
        const ScSlab& slab,
        const ScDirectionSet& dirs,
        const ScOptions& opts,
        const std::vector<f64>& z_centres,
        const std::vector<f64>& row_steps,
        f64 max_step);
};
#endif
