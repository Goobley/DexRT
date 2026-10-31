#ifndef DEXRT_SC_FORMAL_SOLUTION_HPP
#define DEXRT_SC_FORMAL_SOLUTION_HPP
#include "ScDynamicScheduler.hpp"
#include "ScPreparedGeometry.hpp"
#include "ScFormal.hpp"
#include "ScBoundary.hpp"
#include "ScMediumCoefficients.hpp"
#include "ScWavelengthStorage.hpp"
#include <optional>

enum class ScJPath { None, InGamma, Active, Dense };
enum class ScSolveMode { Formal, Prd, Diagnostic };

/// Optional synchronous wall times for disjoint numerical phases of one solve.
struct ScTaskTimings {
    f64 unpack = 0;
    f64 boundary = 0;
    f64 coefficients = 0;
    f64 medium = 0;
    f64 sweep = 0;
    f64 pack = 0;
};

/// Bindings valid only between begin_solve and finish_solve. Task coefficient
/// selection belongs to the cache, not the solve epoch.
struct ScSolveEpoch {
    ScState& state;
    Fp3d emergent;
    bool compute_gamma;
    bool lambda_iterate;
    bool given;
    ScJPath J_path;
    ScBoundary boundary;
    ScSolveMode mode;
    ScTaskTimings timings;
};

/// Numerical ownership at three lifetimes. The scheduler chooses wavelength
/// slot indices and owns independent MPI rows; it never owns these payloads.
struct ScFormalSolution {
    // Run lifetime: immutable geometry and allocation policy.
    ScPreparedGeometry geometry;
    ScOptions opts;
    i32 wave_batch;
    ThermalisedBc thermalised;

    // One synchronous task: medium, solution and boundary/transfer scratch.
    ScBlockBuffers work;
    Fp1d staging;

    // Wavelength sums span directions; coefficient entries have a separate memory bound.
    ScWavelengthStorage wavelengths;
    ScMediumCoefficientCache medium_coefficient_cache;

    // Borrowed solve bindings are released together at finish_solve.
    std::optional<ScSolveEpoch> epoch;
};

/// Allocate reusable numerical state; no solve is borrowed yet.
ScFormalSolution sc_make_formal_solution(
    const ScState& state,
    ScPreparedGeometry geometry,
    const ScOptions& opts,
    i32 wave_batch,
    ThermalisedBc thermalised = {},
    ScSolveMode mode = ScSolveMode::Formal
);
/// Bind solve policy and request the scheduler's task plan, including PRD clipping.
ScTaskPlan sc_begin_solve(
    ScFormalSolution& fs,
    ScState& state,
    const Fp3d& emergent,
    bool compute_gamma,
    bool lambda_iterate,
    ScSolveMode mode = ScSolveMode::Formal
);
/// Reserve wavelength payloads across every direction of an admitted batch.
void sc_open_wavelength_batch(ScFormalSolution& fs, const ScScheduledTask& task, i32 slot);
/// Consume input and finish packing output synchronously. The scheduler can
/// then send that independent host row before calling accumulate.
void sc_sweep(
    ScFormalSolution& fs,
    const ScScheduledTask& task,
    i32 slot,
    const Fp1dHost* input,
    const Fp1dHost* output
);
/// Consume this task's solution before reusing its workspace or cache selection.
void sc_accumulate(ScFormalSolution& fs, const ScScheduledTask& task, i32 slot);
/// Publish angular sums before the scheduler reuses the wavelength slot.
void sc_close_wavelength_batch(ScFormalSolution& fs, const ScScheduledTask& task, i32 slot);
/// Publish final results and release borrowed solve state.
void sc_finish_solve(ScFormalSolution& fs);
#endif
