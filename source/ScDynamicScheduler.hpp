#ifndef DEXRT_SC_DYNAMIC_SCHEDULER_HPP
#define DEXRT_SC_DYNAMIC_SCHEDULER_HPP
#include "Config.hpp"
#include "MpiWrapper.hpp"
#include "Types.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <utility>

enum class ScStream { Up, Down, Local };

/// Array/wire indexing is the only place streams are represented as integers.
constexpr i32 sc_stream_index(ScStream stream) {
    return static_cast<i32>(stream);
}

/// A complete local-slab job. Sequence numbers are
/// ordered within each stream and define message matching between ranks.
struct ScScheduledTask {
    i32 la;    // first wavelength, inclusive
    i32 end;   // last wavelength, exclusive
    i32 batch; // direction batch, interpreted by the executor
    ScStream stream;
    i32 seq;                  // position within this stream
    i32 wavelength_batch_idx; // wavelength-storage lifetime shared by direction tasks
    i64 message_count;
};
struct ScTaskPlan {
    std::vector<ScScheduledTask> tasks;
    i32 num_wavelength_batches = 0;
    i32 num_direction_batches = 0;
    i32 slot_count = 0; // zero: no wavelength storage needs admission
    i64 max_message_count = 0;
    std::vector<i64> layout; // exact cross-rank matching contract
};
struct ScPreparedGeometry;
/// Build ordered streams and the exact cross-rank MPI matching contract.
/// The optional interval clips wavelength batches for a PRD-only solve.
ScTaskPlan sc_make_task_plan(
    const ScPreparedGeometry& geometry,
    i32 nwave,
    i32 wave_batch,
    i32 slot_count,
    std::optional<std::pair<i32, i32>> wavelength_range = {});
struct ScFormalSolution;

/// Chooses ready heads of UP, DOWN and LOCAL FIFO streams. Owns bounded MPI
/// buffers and wavelength-slot admission; the executor owns solver resources.
/// UP/DOWN follow neighbouring ranks. LOCAL holds analytic along-y work in
/// 2D Healpix and fills communication gaps at lower priority.
struct ScDynamicScheduler {
    struct Impl;
    std::unique_ptr<Impl> impl;

    ScDynamicScheduler(const MpiState& mpi, i32 receive_window);
    ~ScDynamicScheduler();
    void run(const ScTaskPlan& plan, ScFormalSolution& executor);
};
#endif
