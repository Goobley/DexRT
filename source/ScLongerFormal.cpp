#include "ScPreparedGeometry.hpp"
#include "ScFormal.hpp"
#include "ScLongerMarch.hpp"
#include "ScLongerIntegration.hpp"
#include "LoopUtils.hpp"

namespace {
/// Map an entry label to its physical face; this is done once per characteristic.
YAKL_INLINE fp_t sc_longer_inflow_value(
    const ScLongerLayout& layout,
    const Fp3d& bottom,
    const Fp3d& side_x,
    const Fp3d& side_y,
    i32 d,
    i32 w,
    ScLongerEntry entry) {
    if (entry.x >= 0 && entry.y >= 0) {
        return bottom(d, w, sc_longer_face_index(layout, entry.x, entry.y));
    }
    if (entry.y < 0) {
        return side_y(d, w, entry.plane * layout.side_rows + sc_longer_line_index(layout, entry.x, entry.y));
    }
    const i64 row = i64(layout.grid.ny) * layout.margin_x;
    return side_x(d, w, entry.plane * row + i64(entry.y) * layout.margin_x + entry.x + layout.margin_x);
}

/// GPU: one lane carries I along a whole characteristic. The DDA and formal
/// integration are visible together, followed by explicit result writes.
template <int Dim, bool UniformZ, bool Loss>
void march_gpu_characteristics(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task) {
    JasUnpack(task.batch, longer, frames);
    JasUnpack(longer, layout);
    JasUnpack(work.medium, chi, S);
    JasUnpack(work.solution, I_ext);
    JasUnpack(work.longer_storage(), bottom, side_x, side_y);
    JasUnpack(work, incoming, outgoing);
    JasUnpack(task, loss);
    const bool has_upstream = task.has_upstream;
    const i64 plane = geometry.grid.plane_size();
    const i64 count = layout.inflow_count;
    dex_parallel_for(
        "SC Longer march characteristics",
        FlatLoop<3>(task.batch.num_dirs, task.wave_batch, count),
        YAKL_LAMBDA(i32 d, i32 w, i64 ray) {
            JasUse(frames); // nvcc requires capture before constexpr-if.
            const fp_t loss_weight = Loss ? loss.weight(d, w) : FP(0.0);
            const auto entry = sc_longer_entry(layout, ray);
            fp_t I = ray < plane && has_upstream
                         ? incoming(d, w, ray)
                         : sc_longer_inflow_value(
                             layout,
                             bottom,
                             side_x,
                             side_y,
                             d,
                             w,
                             entry
                         );
            auto march = sc_begin_longer_grid<Dim, UniformZ>(longer, longer.directions(d), entry);
            while (march.active) {
                const auto segment = march.segment();
                const i64 cell = segment.cell;
                const i64 medium = cell + SC_MEDIUM_HALO * plane;
                const fp_t tau = chi(d, w, medium) * segment.length;

                const auto result = sc_integrate_longer_segment(I, S(d, w, medium), tau, segment.fraction);

                Kokkos::atomic_add(&I_ext(d, w, cell), result.I_ext);
                if constexpr (Loss) {
                    loss.add(
                        frames(d),
                        w,
                        segment.plane,
                        segment.x,
                        segment.y,
                        loss_weight * result.delta_I / segment.plane_length);
                }
                I = result.I;
                sc_step_longer_grid(march);
            }
            if (march.reaches_outflow && march.outgoing >= 0) {
                outgoing(d, w, march.outgoing) = I;
            }
        });
}

/// CPU: keep planes hot and SIMD over characteristics sharing one DDA
/// pattern. Each worker owns a direction/wavelength pair and its cell sums.
template <int Dim, bool UniformZ, bool Loss>
void march_cpu_planes(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task) {
    JasUnpack(task.batch, longer, frames);
    JasUnpack(longer, layout);
    JasUnpack(work.medium, chi, S);
    JasUnpack(work.solution, I_ext, alo, one_minus_alo);
    JasUnpack(work.longer_storage(), bottom, side_x, side_y, host_lines);
    JasUnpack(work, incoming, outgoing);
    JasUnpack(task, loss);
    const bool has_upstream = task.has_upstream;
    const i32 nx = geometry.grid.num_x, ny = geometry.grid.num_y, nz = geometry.grid.num_z;
    const i32 mx = layout.margin_x, my = layout.margin_y;
    const i64 plane = geometry.grid.plane_size();
    dex_parallel_for(
        "SC Longer clear solution",
        FlatLoop<3>(task.batch.num_dirs, task.wave_batch, nz * plane),
        YAKL_LAMBDA(i32 d, i32 w, i64 cell) {
            I_ext(d, w, cell) = alo(d, w, cell) = one_minus_alo(d, w, cell) = FP(0.0);
        });
    // Incoming and outgoing storage double as the two rolling planes.
    dex_parallel_for(
        "SC Longer host planes",
        FlatLoop<2>(task.batch.num_dirs, task.wave_batch),
        YAKL_LAMBDA(i32 d, i32 w) {
            JasUse(frames); // nvcc requires capture before constexpr-if.
            const fp_t loss_weight = Loss ? loss.weight(d, w) : FP(0.0);
            const fp_t* chi_dw = &chi(d, w, 0);
            const fp_t* source_dw = &S(d, w, 0);
            fp_t* external_dw = &I_ext(d, w, 0);
            fp_t* alo_dw = &alo(d, w, 0);
            fp_t* complement_dw = &one_minus_alo(d, w, 0);
            fp_t* current = &incoming(d, w, 0);
            fp_t* next = &outgoing(d, w, 0);
            if (!has_upstream) {
                for (i64 c = 0; c < plane; ++c) {
                    current[c] = bottom(d, w, c);
                }
            }
            fp_t* lines = &host_lines(d, w, 0);
            const auto& direction = longer.directions(d);
            for (i32 p = 0; p < nz; ++p) {
                const auto path = sc_longer_plane_path<UniformZ>(longer, direction, p);
                for (i32 y = -my; y < ny; ++y) {
                    for (i32 x = -mx; x < nx; ++x) {
                        const i64 line = sc_longer_line_index(layout, x, y);
                        lines[line] =
                            x >= 0 && y >= 0
                                ? current[i64(y) * nx + x]
                                : sc_longer_inflow_value(
                                    layout,
                                    bottom,
                                    side_x,
                                    side_y,
                                    d,
                                    w,
                                    {p, x, y}
                                );
                    }
                }
                // Geometry supplies contiguous characteristic-to-cell spans;
                // the SIMD integration loop has no wrapping or clipping.
                auto march = ScLongerPatternMarch<Dim>(path, layout, p);
                do {
                    const fp_t cell_weight = march.fraction();
                    for (i32 y = 0; y < ny; ++y) {
                        const auto spans = sc_longer_cell_spans(march, y);
                        for (i32 run = 0; run < 2; ++run) {
                            const auto span = spans.runs[run];
                            SC_GROUP_SIMD
                            for (i32 i = 0; i < span.count; ++i) {
                                const i64 line = span.line + i;
                                const i64 cell = span.cell + i;
                                const i64 medium = cell + SC_MEDIUM_HALO * plane;
                                const fp_t tau = chi_dw[medium] * cell_weight * path.length;

                                const auto result = sc_integrate_longer_segment(
                                    lines[line],
                                    source_dw[medium],
                                    tau,
                                    cell_weight);

                                external_dw[cell] += result.I_ext;
                                alo_dw[cell] += result.alo;
                                complement_dw[cell] += result.one_minus_alo;
                                if constexpr (Loss) {
                                    loss.add(
                                        frames(d),
                                        w,
                                        p,
                                        span.x + i,
                                        y,
                                        loss_weight * result.delta_I / path.length);
                                }
                                lines[line] = result.I;
                            }
                        }
                    }
                } while (sc_step_longer_pattern(march));
                for (i32 y = 0; y < ny; ++y) {
                    const auto spans = sc_longer_outgoing_spans(march, y);
                    for (i32 run = 0; run < 2; ++run) {
                        const auto span = spans.runs[run];
                        for (i32 i = 0; i < span.count; ++i) {
                            next[span.cell + i] = lines[span.line + i];
                        }
                    }
                }
                fp_t* previous = current;
                current = next;
                next = previous;
            }
            if (current != &outgoing(d, w, 0)) {
                for (i64 c = 0; c < plane; ++c) {
                    outgoing(d, w, c) = current[c];
                }
            }
        });
}

/// GPU response pass: one lane owns one cell and computes its entire ALO.
/// Within a plane, the lattice characteristics are translations of the same
/// path. If that path crosses cells in segments of lengths l0, l1, ..., each
/// cell receives those same lengths from different translated characteristics.
/// We therefore replay one representative plane-path pattern for each cell,
/// using THAT cell's opacity for every segment and ignoring the DDA cell
/// offsets. This enumerates local response contributions; it does not follow
/// radiation through neighbouring cells or trace a characteristic across the
/// slab. The geometry is shared, but the opacity-dependent ALO is not.
/// No incoming intensity is needed, and exclusive cell ownership avoids
/// atomic writes to alo/one_minus_alo during the actual characteristic trace.
template <int Dim, bool UniformZ>
void prepare_gpu_cell_response(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task) {
    JasUnpack(task.batch, longer);
    JasUnpack(work.medium, chi);
    JasUnpack(work.solution, I_ext, alo, one_minus_alo);
    const i64 plane = geometry.grid.plane_size();
    const i32 nz = geometry.grid.num_z;
    dex_parallel_for(
        "SC Longer cell response",
        FlatLoop<3>(task.batch.num_dirs, task.wave_batch, nz * plane),
        YAKL_LAMBDA(i32 d, i32 w, i64 cell) {
            const i32 p = cell / plane;
            const auto path = sc_longer_plane_path<UniformZ>(longer, longer.directions(d), p);
            const fp_t opacity = chi(d, w, cell + SC_MEDIUM_HALO * plane);
            fp_t alo_sum = FP(0.0), one_minus_alo_sum = FP(0.0);
            auto march = ScRayMarch<Dim - 1>(path.x, path.y, path.dx, path.dy);
            do {
                const auto segment = march.segment();
                const auto coefficients = sc_longer_response(opacity * segment.fraction() * path.length);
                alo_sum += segment.fraction() * coefficients.alo;
                one_minus_alo_sum += segment.fraction() * coefficients.one_minus_alo;
            } while (march.step());
            I_ext(d, w, cell) = FP(0.0);
            alo(d, w, cell) = alo_sum;
            one_minus_alo(d, w, cell) = one_minus_alo_sum;
        });
}

/// CPU shares a plane pattern across SIMD lanes; GPU gives each characteristic its own lane.
template <int Dim, bool UniformZ, bool Loss>
void march_characteristics(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task) {
    if constexpr (HostDevSameSpace) {
        march_cpu_planes<Dim, UniformZ, Loss>(geometry, work, task);
    } else {
        prepare_gpu_cell_response<Dim, UniformZ>(geometry, work, task);
        march_gpu_characteristics<Dim, UniformZ, Loss>(geometry, work, task);
    }
    Kokkos::fence();
}
} // namespace

namespace {
/// Select dimension and optional loss output outside the characteristic kernels.
template <bool UniformZ>
void solve_spacing(const ScPreparedGeometry& geometry, const ScBlockBuffers& work, const ScSweepTask& task) {
    sc_dispatch_bool(geometry.grid.dim == 3, [&]<bool ThreeD>() {
        sc_dispatch_bool(task.loss.enabled(), [&]<bool Loss>() {
            march_characteristics<ThreeD ? 3 : 2, UniformZ, Loss>(geometry, work, task);
        });
    });
}
} // namespace

void sc_solve_longer(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task) {
    if (task.batch.longer.uniform_z) {
        solve_spacing<true>(geometry, work, task);
    } else {
        solve_spacing<false>(geometry, work, task);
    }
}
