#if !defined(DEXRT_SC_CONFIG_HPP)
#define DEXRT_SC_CONFIG_HPP

#include "Types.hpp"
#include <string>
#include <stdexcept>
#include <cmath>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

/// Short characteristics options. The method choices are selected at runtime
/// (all variants are compiled); fixed tuning constants are below.

/// Stepan: Stepan et al. 2020 octant tables completed by rotations of pi/2
/// about z (exact to the table's order). StepanMirrored: the same tables
/// completed by mirroring, as Lightweaver does (azimuthal moments no longer
/// exact, but matches Lightweaver's ray set).
enum class ScQuadratureType {
    Stepan,
    StepanMirrored,
    Healpix
};
constexpr const char* ScQuadratureTypeNames[] = {
    "Stepan",
    "StepanMirrored",
    "Healpix"
};

/// PiecewiseLinear: linear S along the step (as Lightweaver).
/// PiecewiseLinearEmissivity: the same with each end's weight scaled by its own
/// opacity (renormalised), so each S is paired with its own chi: exact for
/// linear eta in the thin limit.
enum class ScFormalSolver {
    PiecewiseConstant,
    PiecewiseLinear,
    PiecewiseLinearEmissivity
};
constexpr const char* ScFormalSolverNames[] = {
    "PiecewiseConstant",
    "PiecewiseLinear",
    "PiecewiseLinearEmissivity"
};

enum class ScInterp {
    Linear,
    VanLeer,
    MonotonizedCentral
};
constexpr const char* ScInterpNames[] = {
    "Linear",
    "VanLeer",
    "MonotonizedCentral"
};

enum class ScXBoundary {
    Fixed,
    Periodic
};
constexpr const char* ScXBoundaryNames[] = {
    "Fixed",
    "Periodic"
};

/// Short: characteristics restart each row from an interpolated upwind value.
/// LongerCharacteristics: straight characteristics on a global lattice carry
/// their intensity from row to row (no upwind interpolation); cell-mean output.
enum class ScCharacteristics {
    Short,
    LongerCharacteristics
};
constexpr const char* ScCharacteristicsNames[] = {
    "Short",
    "LongerCharacteristics"
};

/// Off: PRD lines are treated in CRD. Hybrid: resonance PRD in the hybrid
/// approximation (comoving-frame angle-averaged redistribution; Leenaarts et
/// al. 2012, Sukhorukov & Leenaarts 2016), coherence in the impact limit.
enum class ScPrdMode {
    Off,
    Hybrid
};
constexpr const char* ScPrdModeNames[] = {
    "Off",
    "Hybrid"
};

struct ScOptions {
    /// Independent medium-coefficient cache entries: these large payloads have
    /// a separate memory bound from the admitted wavelength sums.
    i32 base_slots = 2;
    /// Send/receive ring rows per communicating MPI stream.
    i32 receive_window = 4;
    /// Wavelengths per task (0: choose collectively).
    i32 wave_batch = 0;
    /// Admitted wavelength batches holding J/loss/previous-J across directions
    /// (0: ranks + receive_window, capped by the number of batches).
    i32 j_slots = 0;
    ScQuadratureType quadrature = ScQuadratureType::Stepan;
    /// Stepan: the quadrature order (3, 7, 9, 11, 13, 15). Healpix: the order
    /// (nside = 2**order).
    int quadrature_order = 7;
    ScFormalSolver formal_solver = ScFormalSolver::PiecewiseLinearEmissivity;
    ScInterp interpolation = ScInterp::MonotonizedCentral;
    ScXBoundary x_boundary = ScXBoundary::Fixed;
    ScXBoundary y_boundary = ScXBoundary::Fixed; /// 3D only
    ScCharacteristics characteristics = ScCharacteristics::LongerCharacteristics;
    /// Promweaver boundary: average the incident intensity of each direction
    /// over its equal-area cone (solid angle 4 pi w), with boundary_cone_mu
    /// Gauss-Legendre nodes in cos(angle from the axis) and boundary_cone_phi
    /// azimuths. 0: sample along the direction only.
    int boundary_cone_mu = 4;
    int boundary_cone_phi = 8;
    ScPrdMode prd = ScPrdMode::Off;
    /// Overrides: PRD sub-iterations per iteration, their tolerance on the
    /// relative change of rho, and the first iteration with rho != 1.
    i32 prd_subiterations = 3;
    f64 prd_tolerance = 1e-3;
    i32 prd_start_iter = 0;
};

/// PRD sub-iterations before the tolerance can end them: a single one leaves a
/// slow mode (~1e-3) in the populations (as in Lightweaver).
constexpr int SC_PRD_MIN_SUBITERATIONS = 2;

/// Rays crossing more columns than this per row (plane) step are rejected (a
/// cost cap; laps of a periodic axis are swept). Rays with mu_z = 0 are
/// rejected too (they would need an in-plane solver; Stepan has none, and
/// Healpix is rotated off them).
constexpr f64 SC_MAX_ROW_CROSSINGS = 1000.0;
/// Number of directions solved (and stored) together.
constexpr int SC_DIR_BATCH = 12;
/// Fraction of the memory pool the per-batch working set may use when choosing
/// how many wavelengths to solve together.
constexpr double SC_WAVE_BATCH_MEM_FRACTION = 0.6;
/// Threads per (direction, wavelength) team in the sweep (on GPUs): narrow
/// rows prefer smaller teams (several resident per SM hide the per-row
/// barriers), wide rows larger ones.
constexpr int SC_SWEEP_TEAM_SIZE = 256;
constexpr int SC_SWEEP_TEAM_SIZE_WIDE = 512;
constexpr int SC_SWEEP_WIDE_ROW = 1024;

inline int sc_sweep_team_size(int num_x) {
    return (num_x >= SC_SWEEP_WIDE_ROW) ? SC_SWEEP_TEAM_SIZE_WIDE : SC_SWEEP_TEAM_SIZE;
}
/// Number of canonical rows of medium halo kept either side of the slab
/// (also the halo rows of the ScCells index space).
constexpr i32 SC_MEDIUM_HALO = 2;

/// Cells handled per thread in the per-cell ray loops (Gamma, medium). On
/// CPU two cache lines of f32, so each (direction, wavelength) read is whole
/// lines (one cell at a time strides across every direction and wavelength
/// array) and the per-ray setup is shared by more cells; on GPU one,
/// neighbouring threads coalescing instead.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
constexpr int SC_CELL_GROUP = 1;
#define SC_GROUP_SIMD
#else
constexpr int SC_CELL_GROUP = 32;
/// Vectorise a loop over a cell group (host only).
#define SC_GROUP_SIMD _Pragma("omp simd")
#endif

/// Prefetch the cache line at p (host only).
YAKL_INLINE void sc_prefetch(const void* p) {
#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
    __builtin_prefetch(p);
#endif
}
/// Gamma evaluates the line profiles from the per-cell line coefficients of
/// ScMediumCoefficients (saving the broadening's pow per ray) on CPU. On GPU this was 5x
/// slower (register pressure), so the coefficients are recomputed there.
constexpr bool SC_GAMMA_CACHED_LINES = SC_CELL_GROUP > 1;
/// Longer characteristics: segments of a row processed per pass (bounds the
/// segment scratch; rays crossing more columns take several passes).
constexpr int SC_LONGER_SEG_BLOCK = 32;
/// Preferred tasks per remote stream per rank, so the pipeline fill/drain is
/// a small fraction of each formal solution.
constexpr int SC_PIPELINE_TASKS_PER_RANK = 8;
/// Relative cost per row cell (sweep) and per active cell (medium, Gamma,
/// J) [ns/node/dir/wave], weighting the z-slab partition.
constexpr f64 SC_SWEEP_COST = 14.0;
constexpr f64 SC_ACTIVE_COST = 25.0;
/// 3D activity tiles (as dexrt_3d's blocks).
constexpr int SC_TILE_3D = BLOCK_SIZE_3D;
/// |mu_x|, |mu_z| below which a 2D direction is treated as along y.
constexpr fp_t SC_DEGENERATE_MU = FP(1e-6);

template <typename E, int N>
E sc_parse_enum(const YAML::Node& node, const char* key, const char* const (&names)[N], E def) {
    if (!node[key]) {
        return def;
    }
    std::string v = node[key].as<std::string>();
    for (int i = 0; i < N; ++i) {
        if (v == names[i]) {
            return E(i);
        }
    }
    throw std::runtime_error(fmt::format("Unexpected value for sc.{}: {}", key, v));
}

inline ScOptions parse_sc_options(const YAML::Node& file) {
    ScOptions opts{};
    if (!file["sc"] || !file["sc"].IsMap()) {
        return opts;
    }
    const auto& sc = file["sc"];
    if (sc["base_slots"]) {
        opts.base_slots = sc["base_slots"].as<i32>();
    }
    if (sc["receive_window"]) {
        opts.receive_window = sc["receive_window"].as<i32>();
    }
    if (sc["wave_batch"]) {
        opts.wave_batch = sc["wave_batch"].as<i32>();
    }
    if (sc["j_slots"]) {
        opts.j_slots = sc["j_slots"].as<i32>();
    }
    if (opts.base_slots < 1 || opts.receive_window < 1 || opts.wave_batch < 0 || opts.j_slots < 0) {
        throw std::runtime_error("sc.base_slots and sc.receive_window must be >= 1, sc.wave_batch and sc.j_slots >= 0");
    }

    opts.quadrature = sc_parse_enum(sc, "quadrature", ScQuadratureTypeNames, opts.quadrature);
    if (opts.quadrature == ScQuadratureType::Healpix) {
        opts.quadrature_order = 1;
    }
    if (sc["quadrature_order"] && !sc["quadrature_order"].IsNull()) {
        opts.quadrature_order = sc["quadrature_order"].as<int>();
    }
    opts.formal_solver = sc_parse_enum(sc, "formal_solver", ScFormalSolverNames, opts.formal_solver);
    opts.interpolation = sc_parse_enum(sc, "interpolation", ScInterpNames, opts.interpolation);
    opts.x_boundary = sc_parse_enum(sc, "x_boundary", ScXBoundaryNames, opts.x_boundary);
    opts.y_boundary = sc_parse_enum(sc, "y_boundary", ScXBoundaryNames, opts.y_boundary);
    opts.characteristics = sc_parse_enum(sc, "characteristics", ScCharacteristicsNames, opts.characteristics);
    if (sc["boundary_cone_mu"]) {
        opts.boundary_cone_mu = sc["boundary_cone_mu"].as<int>();
    }
    if (sc["boundary_cone_phi"]) {
        opts.boundary_cone_phi = sc["boundary_cone_phi"].as<int>();
    }
    if (opts.boundary_cone_mu < 0 || opts.boundary_cone_phi < 1) {
        throw std::runtime_error("sc.boundary_cone_mu must be >= 0 and sc.boundary_cone_phi >= 1");
    }
    opts.prd = sc_parse_enum(sc, "prd", ScPrdModeNames, opts.prd);
    if (sc["prd_subiterations"]) {
        opts.prd_subiterations = sc["prd_subiterations"].as<i32>();
    }
    if (sc["prd_tolerance"]) {
        opts.prd_tolerance = sc["prd_tolerance"].as<f64>();
    }
    if (sc["prd_start_iter"]) {
        opts.prd_start_iter = sc["prd_start_iter"].as<i32>();
    }
    if (opts.prd_subiterations < 0 || opts.prd_start_iter < 0) {
        throw std::runtime_error("sc.prd_subiterations and sc.prd_start_iter must be >= 0");
    }
    return opts;
}

#else
#endif
