#include "ScDynamicScheduler.hpp"
#include "ScFormalSolution.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <fmt/core.h>
#include <limits>

namespace {
using Clock = std::chrono::steady_clock;
f64 seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<f64>(b - a).count();
}
} // namespace
ScTaskPlan sc_make_task_plan(
    const ScPreparedGeometry& geometry,
    i32 nwave,
    i32 wave_batch,
    i32 slot_count,
    std::optional<std::pair<i32, i32>> wavelength_range
) {
    ScTaskPlan plan;
    plan.num_wavelength_batches = (nwave + wave_batch - 1) / wave_batch;
    plan.num_direction_batches = geometry.batches.size();
    plan.slot_count = slot_count;
    plan.max_message_count =
        i64(sc_max_batch_dirs(geometry.batches)) * wave_batch * geometry.grid.plane_size();
    std::array<i32, 3> sequence{};
    for (i32 la = 0; la < nwave; la += wave_batch) {
        i32 start = la;
        i32 end = std::min(la + wave_batch, nwave);
        if (wavelength_range) {
            start = std::max(start, wavelength_range->first);
            end = std::min(end, wavelength_range->second);
            if (start >= end) {
                continue;
            }
        }
        for (i32 d = 0; d < geometry.batches.size(); ++d) {
            const auto& batch = geometry.batches[d];
            ScStream stream = ScStream::Local;
            if (batch.mode == ScRayMode::Sweep) {
                stream = batch.flip_z ? ScStream::Down : ScStream::Up;
            }
            plan.tasks.push_back(ScScheduledTask{
                .la = start,
                .end = end,
                .batch = d,
                .stream = stream,
                .seq = sequence[sc_stream_index(stream)]++,
                .wavelength_batch_idx = la / wave_batch,
                .message_count =
                    i64(geometry.batches[d].num_dirs) * (end - start) * geometry.grid.plane_size(),
            });
        }
    }
    // Exact MPI matching contract, including direction membership and PRD
    // clipping. The scheduler adds its receive-window value before comparing.
    plan.layout = {
        wave_batch,
        geometry.dirs.num_dirs,
        i64(geometry.batches.size()),
        geometry.grid.num_x,
        geometry.grid.num_y,
        geometry.slab.global_num_z,
        nwave,
        wavelength_range.has_value(),
    };
    for (const auto& batch : geometry.batches) {
        plan.layout.push_back(i32(batch.mode));
        plan.layout.push_back(batch.flip_z);
        plan.layout.push_back(batch.num_dirs);
        plan.layout.push_back(batch.dir_idx_host.size());
        for (auto dir : batch.dir_idx_host) {
            plan.layout.push_back(dir);
        }
    }

    for (const auto& task : plan.tasks) {
        plan.layout.insert(
            plan.layout.end(),
            {task.la, task.end, task.batch, sc_stream_index(task.stream), task.seq});
    }

    return plan;
}

namespace {
/// One ordered MPI stream. Only its head is eligible; different streams
/// may advance at different rates as inputs and storage become available.
struct Stream {
    std::vector<i32> tasks;
    i32 head = 0;

    bool empty() const { return head == tasks.size(); }
    i32 front() const { return tasks[head]; }
    void complete() { ++head; }
};

/// Per-solve diagnostics, independent of task and communication-buffer ownership.
struct ScSchedulerStats {
    std::array<i64, 3> tasks{};
    std::array<f64, 3> busy{};
    std::array<f64, 3> max_task{};
    std::array<i64, 8> waits{};
    std::array<f64, 8> wait_seconds{};
    i64 sent_bytes = 0;
    i64 received_bytes = 0;
    i32 peak_wavelength_slots = 0;
    f64 setup = 0;
    f64 open = 0;
    f64 accumulate = 0;
    f64 close = 0;
    f64 drain = 0;
    f64 finish = 0;
};

/// Numerical calls may enqueue GPU work; detailed timings measure completion.
template <class Run>
void sc_scheduler_measure(bool enabled, f64& total, const Run& run) {
    if (!enabled) {
        run();
        return;
    }
    Kokkos::fence();
    const auto start = Clock::now();
    run();
    Kokkos::fence();
    total += seconds(start, Clock::now());
}

/// Collect small text reports so ranks print contiguously and in rank order.
void sc_print_rank_reports(const MpiState& mpi, const std::string& report) {
    std::string reports = report;
#ifdef HAVE_MPI
    if (mpi.num_ranks > 1) {
        const int size = report.size();
        std::vector<int> sizes(mpi.num_ranks), offsets(mpi.num_ranks);
        MPI_Gather(
            &size,
            1,
            MPI_INT,
            sizes.data(),
            1,
            MPI_INT,
            0,
            mpi.comm
        );
        if (mpi.rank == 0) {
            int total = 0;
            for (int rank = 0; rank < mpi.num_ranks; ++rank) {
                offsets[rank] = total;
                total += sizes[rank];
            }
            reports.resize(total);
        }
        MPI_Gatherv(
            report.data(),
            size,
            MPI_CHAR,
            reports.data(),
            sizes.data(),
            offsets.data(),
            MPI_CHAR,
            0,
            mpi.comm
        );
    }
#endif
    if (mpi.rank == 0) {
        fmt::print(
            "SC scheduler detailed stats (synchronous phase wall times, seconds):\n"
            "  Wait labels combine blockers across FIFO heads; phase times are subsets of busy.\n{}",
            reports
        );
    }
}

/// Wait masks describe simultaneous FIFO-head constraints, not MPI transfer latency.
std::string sc_scheduler_report(
    const MpiState& mpi,
    const ScTaskPlan& plan,
    const ScFormalSolution& executor,
    const ScTaskTimings& phases,
    const ScSchedulerStats& stats,
    i64 active_cells,
    f64 elapsed,
    f64 busy,
    f64 idle
) {
    const auto& slab = executor.geometry.slab;
    const auto& cache = executor.medium_coefficient_cache;
    const f64 wait = idle - stats.drain;
    const f64 overhead = std::max(0.0, elapsed - stats.setup - busy - idle - stats.finish);
    std::string report = fmt::format(
        "  SC scheduler rank {}: elapsed={:.6f} busy={:.6f} wait={:.6f} drain={:.6f} "
        "setup={:.6f} finish={:.6f} overhead={:.6f}\n"
        "    slab=[{}, {}) active_cells={} tasks={} wavelength_slots_peak={}/{}\n",
        mpi.rank,
        elapsed,
        busy,
        wait,
        stats.drain,
        stats.setup,
        stats.finish,
        overhead,
        slab.z_start,
        slab.z_end,
        active_cells,
        plan.tasks.size(),
        stats.peak_wavelength_slots,
        plan.slot_count
    );
    constexpr const char* names[] = {"Up", "Down", "Local"};
    for (i32 stream = 0; stream < 3; ++stream) {
        report += fmt::format(
            "    stream {}: tasks={} busy={:.6f} mean_task={:.6f} max_task={:.6f}\n",
            names[stream],
            stats.tasks[stream],
            stats.busy[stream],
            stats.tasks[stream] ? stats.busy[stream] / stats.tasks[stream] : 0.0,
            stats.max_task[stream]
        );
    }
    report += fmt::format(
        "    phases: open={:.6f} unpack={:.6f} boundary={:.6f} coefficients={:.6f} "
        "medium={:.6f} sweep={:.6f} pack={:.6f} accumulate={:.6f} close={:.6f}\n"
        "    cache: entries={} builds={} hits={} ; MPI payload: sent_MiB={:.3f} received_MiB={:.3f}\n",
        stats.open,
        phases.unpack,
        phases.boundary,
        phases.coefficients,
        phases.medium,
        phases.sweep,
        phases.pack,
        stats.accumulate,
        stats.close,
        cache.entries.size(),
        cache.builds,
        cache.hits,
        stats.sent_bytes / 1048576.0,
        stats.received_bytes / 1048576.0
    );
    constexpr const char* masks[] = {
        "none",
        "input",
        "send",
        "input+send",
        "slot",
        "input+slot",
        "send+slot",
        "input+send+slot",
    };
    for (i32 mask = 0; mask < 8; ++mask) {
        if (stats.waits[mask]) {
            report += fmt::format(
                "    wait {}: calls={} seconds={:.6f}\n",
                masks[mask],
                stats.waits[mask],
                stats.wait_seconds[mask]
            );
        }
    }
    return report;
}

/// A wavelength batch owns its slot from its first direction through its
/// last. Coefficient-cache eviction is independent of this lifetime.
struct WavelengthSlots {
    std::vector<i32> slot_of;
    std::vector<i32> remaining;
    std::vector<i32> free;

    WavelengthSlots(i32 waves, i32 directions, i32 slots) : slot_of(waves, -1), remaining(waves, directions) {
        for (i32 i = slots; i > 0; --i) {
            free.push_back(i - 1);
        }
    }

    bool available(i32 wave) const { return slot_of[wave] >= 0 || !free.empty(); }
    bool is_open(i32 wave) const { return slot_of[wave] >= 0; }
    i32 slot(i32 wave) const { return slot_of[wave]; }
    /// Reserve a free payload index until the final direction of this batch.
    i32 open(i32 wave) {
        const i32 result = free.back();
        free.pop_back();
        slot_of[wave] = result;
        return result;
    }

    bool last_direction(i32 wave) { return --remaining[wave] == 0; }
    /// Only the admission mapping changes; the executor owns the payload.
    void release(i32 wave) {
        free.push_back(slot_of[wave]);
        slot_of[wave] = -1;
    }
};

// Rows in the per-stream receive and send rings, indexed by sequence
// modulo W. A send row stays owned by MPI until its request completes.
struct Row {
    Fp1dHost data;
    i32 seq = -1;
    bool ready = false;
#ifdef HAVE_MPI
    MPI_Request request = MPI_REQUEST_NULL;
#endif
};
/// Host buffers outlive the synchronous compute workspace. A received
/// buffer can be reposted only after unpack/sweep; a sent buffer remains
/// owned by MPI until completion. This object owns both lifetimes.
struct Transport {
    std::array<std::vector<Row>, 2> rows;
    std::array<std::vector<Row>, 2> sends;
#ifdef HAVE_MPI
    MPI_Comm comm[2] = {MPI_COMM_NULL, MPI_COMM_NULL};
    // Reused scratch for request polling; row buffers retain ownership.
    std::vector<Row*> owners;
    std::vector<MPI_Request> handles;
    std::vector<int> done;
#endif

    /// A ring position is usable only after the matching receive has completed.
    bool input_ready(const ScScheduledTask& task, i32 window) const {
        if (task.stream == ScStream::Local || rows[sc_stream_index(task.stream)].empty()) {
            return true;
        }
        const auto& row = rows[sc_stream_index(task.stream)][task.seq % window];
        return row.seq == task.seq && row.ready;
    }

    /// Do not overwrite a packed row while MPI still borrows it.
    bool send_available(const ScScheduledTask& task, i32 window) const {
#ifdef HAVE_MPI
        if (task.stream != ScStream::Local && !sends[sc_stream_index(task.stream)].empty()) {
            return sends[sc_stream_index(task.stream)][task.seq % window].request == MPI_REQUEST_NULL;
        }
#endif
        return true;
    }

    /// Reserve the sequence-tagged receive row until sweep consumes it.
    void post_input(const ScScheduledTask& task, i64 count, i32 source, i32 window) {
        if (task.stream == ScStream::Local || rows[sc_stream_index(task.stream)].empty()) {
            return;
        }
        auto& row = rows[sc_stream_index(task.stream)][task.seq % window];
        row.seq = task.seq;
        row.ready = false;
#ifdef HAVE_MPI
        MPI_Irecv(
            row.data.data(),
            count,
            get_FpMpi(),
            source,
            task.seq % window,
            comm[sc_stream_index(task.stream)],
            &row.request
        );
#endif
    }

    const Fp1dHost& input(const ScScheduledTask& task, i32 window) const {
        return rows[sc_stream_index(task.stream)][task.seq % window].data;
    }

    Fp1dHost& output(const ScScheduledTask& task, i32 window) {
        return sends[sc_stream_index(task.stream)][task.seq % window].data;
    }

    /// The executor finished packing; MPI now borrows this host row until completion.
    void send_output(const ScScheduledTask& task, i32 destination, i32 window) {
#ifdef HAVE_MPI
        auto& row = sends[sc_stream_index(task.stream)][task.seq % window];
        MPI_Isend(
            row.data.data(),
            task.message_count,
            get_FpMpi(),
            destination,
            task.seq % window,
            comm[sc_stream_index(task.stream)],
            &row.request
        );
#endif
    }

    /// Complete requests before reusing their buffers. Receive completion
    /// makes input available; send completion relinquishes MPI ownership.
    bool progress(bool wait) {
#ifdef HAVE_MPI
        owners.clear();
        handles.clear();
        for (auto* ring : {&rows, &sends}) {
            for (auto& stream : *ring) {
                for (auto& row : stream) {
                    if (row.request != MPI_REQUEST_NULL) {
                        handles.push_back(row.request);
                        owners.push_back(&row);
                    }
                }
            }
        }
        if (handles.empty()) {
            return false;
        }
        int count = 0;
        done.resize(handles.size());
        if (wait) {
            MPI_Waitany(handles.size(), handles.data(), done.data(), MPI_STATUS_IGNORE);
            count = 1;
        } else {
            MPI_Testsome(
                handles.size(),
                handles.data(),
                &count,
                done.data(),
                MPI_STATUSES_IGNORE
            );
        }
        for (i32 i = 0; i < handles.size(); ++i) {
            owners[i]->request = handles[i];
        }
        for (i32 i = 0; i < count; ++i) {
            owners[done[i]]->ready = true;
        }
        return true;
#else
        return false;
#endif
    }

    /// Drain sends before transport destruction; all receives must already be consumed.
    void finish() {
#ifdef HAVE_MPI
        for (auto& stream : sends) {
            for (auto& row : stream) {
                MPI_Wait(&row.request, MPI_STATUS_IGNORE);
            }
        }
        for (auto& stream : rows) {
            for (auto& row : stream) {
                if (row.request != MPI_REQUEST_NULL) {
                    throw std::runtime_error("SC dynamic termination with an outstanding receive");
                }
            }
        }
#endif
    }

    /// Release requests and communicators, including exceptional task termination.
    ~Transport() {
#ifdef HAVE_MPI
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            // Also drain requests if task execution exits exceptionally.
            for (auto* ring : {&rows, &sends}) {
                for (auto& stream : *ring) {
                    for (auto& row : stream) {
                        if (row.request != MPI_REQUEST_NULL) {
                            MPI_Cancel(&row.request);
                            MPI_Wait(&row.request, MPI_STATUS_IGNORE);
                        }
                    }
                }
            }
            for (auto& context : comm) {
                if (context != MPI_COMM_NULL) {
                    MPI_Comm_free(&context);
                }
            }
        }
#endif
    }
};
} // namespace

struct ScDynamicScheduler::Impl {
    // Communication owns its rings and outstanding MPI requests.
    Transport transport;
    const MpiState mpi;
    const i32 window;

    /// Isolate UP/DOWN message matching and validate the bounded tag space.
    Impl(const MpiState& mpi_, i32 window_) : mpi(mpi_), window(window_) {
        if (window < 1) {
            throw std::runtime_error("SC receive window must be positive");
        }
#ifdef HAVE_MPI
        if (mpi.num_ranks > 1) {
            for (i32 stream = 0; stream < 2; ++stream) {
                MPI_Comm_dup(mpi.comm, &transport.comm[stream]);
                int* limit = nullptr;
                int flag = 0;
                MPI_Comm_get_attr(transport.comm[stream], MPI_TAG_UB, &limit, &flag);
                if (!flag || i64(window) + 1 > *limit) {
                    throw std::runtime_error("SC receive window exceeds MPI tag range");
                }
            }
        }
#endif
    }

#ifdef HAVE_MPI
    /// Message counts, stream order and tag reuse must agree before any rank
    /// posts receives. Compare the complete small description with rank 0,
    /// then report disagreement collectively so every rank takes the same exit.
    void validate_task_layout(std::vector<i64> layout, bool detailed) const {
        if (mpi.num_ranks == 1) {
            return;
        }
        layout.push_back(window);
        layout.push_back(detailed);
        // Broadcast the size first: disagreeing quadratures can produce
        // different numbers of batches and directions on different ranks.
        i64 reference_size = layout.size();
        MPI_Bcast(
            &reference_size,
            1,
            MPI_INT64_T,
            0,
            mpi.comm
        );
        if (reference_size > std::numeric_limits<int>::max()) {
            throw std::runtime_error("SC task layout exceeds MPI count limit");
        }
        std::vector<i64> reference(reference_size);
        if (mpi.rank == 0) {
            reference = layout;
        }
        MPI_Bcast(
            reference.data(),
            i32(reference_size),
            MPI_INT64_T,
            0,
            mpi.comm
        );
        i32 mismatch = layout != reference;
        MPI_Allreduce(
            MPI_IN_PLACE,
            &mismatch,
            1,
            MPI_INT,
            MPI_MAX,
            mpi.comm
        );
        if (mismatch) {
            throw std::runtime_error("SC ranks disagree on task boundaries or stream membership, or sc.scheduler_stats");
        }
    }
#endif

    bool has_upstream(ScStream stream) const {
        return stream == ScStream::Up ? mpi.rank > 0
                                      : stream == ScStream::Down && mpi.rank + 1 < mpi.num_ranks;
    }

    bool has_downstream(ScStream stream) const {
        return stream == ScStream::Up ? mpi.rank + 1 < mpi.num_ranks
                                      : stream == ScStream::Down && mpi.rank > 0;
    }

    i32 downstream_distance(ScStream stream) const {
        return stream == ScStream::Up     ? mpi.num_ranks - 1 - mpi.rank
               : stream == ScStream::Down ? mpi.rank
                                          : -1;
    }

    /// Agree on the task contract, then allocate independent send/receive rings.
    void prepare_transport(
        const ScTaskPlan& plan,
        const std::array<Stream, 3>& streams,
        bool detailed
    ) {
#ifdef HAVE_MPI
        validate_task_layout(plan.layout, detailed);
#endif
        if (plan.max_message_count > std::numeric_limits<int>::max()) {
            throw std::runtime_error("SC intensity row exceeds MPI count limit");
        }
        for (i32 stream = 0; stream < 2; ++stream) {
            if (streams[stream].empty()) {
                continue;
            }
            auto allocate = [&](std::vector<Row>& ring) {
                ring.resize(window);
                for (auto& row : ring) {
                    if (!row.data.initialized() || row.data.extent(0) != plan.max_message_count) {
                        row.data = Fp1dHost("sc_dynamic_message", plan.max_message_count);
                    }
                }
            };
            if (has_upstream(static_cast<ScStream>(stream))) {
                allocate(transport.rows[stream]);
            }
            if (has_downstream(static_cast<ScStream>(stream))) {
                allocate(transport.sends[stream]);
            }
        }
    }

    /// Post the next task in this stream only when its upstream rank exists.
    void
    post_receive(
        ScStream kind,
        i32 sequence,
        const ScTaskPlan& plan,
        const std::array<Stream,
        3>& streams
    ) {
        const i32 stream = sc_stream_index(kind);
        if (sequence >= streams[stream].tasks.size() || !has_upstream(kind)) {
            return;
        }
        const auto& task = plan.tasks[streams[stream].tasks[sequence]];
        transport
            .post_input(task, task.message_count, kind == ScStream::Up ? mpi.rank - 1 : mpi.rank + 1, window);
    }

    /// Consider at most three heads. FIFO order fixes MPI matching; availability
    /// Bits identify unmet prerequisites; several may block the same FIFO head.
    i32 blocked_by(
        const ScTaskPlan& plan,
        const ScScheduledTask& task,
        const WavelengthSlots& wavelengths
    ) const {
        i32 mask = 0;
        if (!transport.input_ready(task, window)) {
            mask |= 1;
        }
        if (!transport.send_available(task, window)) {
            mask |= 2;
        }
        if (plan.slot_count && !wavelengths.available(task.wavelength_batch_idx)) {
            mask |= 4;
        }
        return mask;
    }

    /// chooses between streams. Prefer feeding the longest downstream pipeline.
    i32 choose_ready(
        const ScTaskPlan& plan,
        const std::array<Stream, 3>& streams,
        const WavelengthSlots& wavelengths
    ) const {
        i32 chosen = -1;
        for (i32 stream = 0; stream < 3; ++stream) {
            if (streams[stream].empty()) {
                continue;
            }
            const i32 id = streams[stream].front();
            const auto& task = plan.tasks[id];
            if (blocked_by(plan, task, wavelengths)) {
                continue;
            }
            if (chosen < 0 || downstream_distance(static_cast<ScStream>(stream)) >
                                  downstream_distance(plan.tasks[chosen].stream)) {
                chosen = id;
            }
        }
        return chosen;
    }

    /// Execute ready FIFO heads and retire each wavelength after its final direction.
    void run(const ScTaskPlan& plan, ScFormalSolution& executor) {
        const bool detailed = executor.opts.scheduler_stats;
        const auto solve_start = Clock::now();
        ScSchedulerStats stats;
        std::array<Stream, 3> streams;
        for (i32 id = 0; id < plan.tasks.size(); ++id) {
            streams[sc_stream_index(plan.tasks[id].stream)].tasks.push_back(id);
        }
        prepare_transport(plan, streams, detailed);
        WavelengthSlots wavelengths(plan.num_wavelength_batches, plan.num_direction_batches, plan.slot_count);
        for (i32 stream = 0; stream < 2; ++stream) {
            for (i32 sequence = 0; sequence < std::min<i32>(window, streams[stream].tasks.size());
                 ++sequence) {
                post_receive(static_cast<ScStream>(stream), sequence, plan, streams);
            }
        }
        if (detailed) {
            stats.setup = seconds(solve_start, Clock::now());
        }
        i64 completed = 0;
        f64 busy = 0, idle = 0;
        while (completed < plan.tasks.size()) {
            const bool pending = transport.progress(false);
            const i32 id = choose_ready(plan, streams, wavelengths);
            if (id < 0) {
                if (!pending) {
                    throw std::runtime_error(fmt::format(
                        "SC dynamic stalled on rank {}: heads {}/{}/{}, completed {}/{}",
                        mpi.rank,
                        streams[0].head,
                        streams[1].head,
                        streams[2].head,
                        completed,
                        plan.tasks.size()
                    ));
                }
                i32 mask = 0;
                if (detailed) {
                    for (const auto& stream : streams) {
                        if (!stream.empty()) {
                            mask |= blocked_by(plan, plan.tasks[stream.front()], wavelengths);
                        }
                    }
                }
                const auto start = Clock::now();
                transport.progress(true);
                const f64 waited = seconds(start, Clock::now());
                idle += waited;
                if (detailed) {
                    ++stats.waits[mask];
                    stats.wait_seconds[mask] += waited;
                }
                continue;
            }
            const auto start = Clock::now();
            const auto& task = plan.tasks[id];
            i32 slot = -1;
            if (plan.slot_count) {
                if (!wavelengths.is_open(task.wavelength_batch_idx)) {
                    slot = wavelengths.open(task.wavelength_batch_idx);
                    sc_scheduler_measure(detailed, stats.open, [&] {
                        sc_open_wavelength_batch(executor, task, slot);
                    });
                    if (detailed) {
                        stats.peak_wavelength_slots = std::max(
                            stats.peak_wavelength_slots,
                            plan.slot_count - i32(wavelengths.free.size())
                        );
                    }
                } else {
                    slot = wavelengths.slot(task.wavelength_batch_idx);
                }
            }
            const Fp1dHost* input = has_upstream(task.stream) ? &transport.input(task, window) : nullptr;
            const Fp1dHost* output = has_downstream(task.stream) ? &transport.output(task, window) : nullptr;
            sc_sweep(
                executor,
                task,
                slot,
                input,
                output
            );
            // Sweep has consumed input; its ring position can receive the next
            // window now, independently of accumulation-slot availability.
            if (input) {
                post_receive(task.stream, task.seq + window, plan, streams);
            }
            if (output) {
                // Sweep completed the copy into the send ring. MPI retains
                // only that row until completion; task/slot storage can be reused.
                transport.send_output(
                    task,
                    task.stream == ScStream::Up ? mpi.rank + 1 : mpi.rank - 1,
                    window
                );
            }
            // Start downstream communication before local reductions.
            sc_scheduler_measure(detailed, stats.accumulate, [&] {
                sc_accumulate(executor, task, slot);
            });
            if (plan.slot_count && wavelengths.last_direction(task.wavelength_batch_idx)) {
                sc_scheduler_measure(detailed, stats.close, [&] {
                    sc_close_wavelength_batch(executor, task, slot);
                });
                wavelengths.release(task.wavelength_batch_idx);
            }
            streams[sc_stream_index(task.stream)].complete();
            ++completed;
            const f64 duration = seconds(start, Clock::now());
            busy += duration;
            if (detailed) {
                const i32 stream = sc_stream_index(task.stream);
                ++stats.tasks[stream];
                stats.busy[stream] += duration;
                stats.max_task[stream] = std::max(stats.max_task[stream], duration);
                stats.sent_bytes += output ? task.message_count * sizeof(fp_t) : 0;
                stats.received_bytes += input ? task.message_count * sizeof(fp_t) : 0;
            }
        }
        const auto start = Clock::now();
        transport.finish();
        stats.drain = seconds(start, Clock::now());
        idle += stats.drain;
        const auto phases = executor.epoch.value().timings;
        const i64 active_cells = executor.epoch.value().state.cells.num_active;
        sc_scheduler_measure(detailed, stats.finish, [&] {
            sc_finish_solve(executor);
        });
        const f64 elapsed = seconds(solve_start, Clock::now());
        if (mpi.rank == 0) {
            fmt::print(
                "SC dynamic: {} tasks/rank, rank-0 busy {:.4f}s idle {:.4f}s\n",
                plan.tasks.size(),
                busy,
                idle
            );
        }
        if (detailed) {
            sc_print_rank_reports(
                mpi,
                sc_scheduler_report(
                    mpi,
                    plan,
                    executor,
                    phases,
                    stats,
                    active_cells,
                    elapsed,
                    busy,
                    idle
                )
            );
        }
    }
};
ScDynamicScheduler::ScDynamicScheduler(const MpiState& mpi, i32 receive_window)
    : impl(std::make_unique<Impl>(mpi, receive_window)) {}
ScDynamicScheduler::~ScDynamicScheduler() = default;
void ScDynamicScheduler::run(const ScTaskPlan& plan, ScFormalSolution& executor) {
    impl->run(plan, executor);
}
