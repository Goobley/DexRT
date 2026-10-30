#include "ScPreparedGeometry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "ScLongerGeometry.hpp"

namespace {
/// Derive every named array extent once; allocation and budgeting consume these counts.
ScShortInflowSizes short_inflow_sizes(const ScGrid& grid, const ScStorageRequirements& storage) {
    const i64 nx = grid.num_x, ny = grid.num_y, np = grid.num_z;
    if (grid.dim == 2) {
        return {
            .bottom = nx,
            .edge = np,
            .ghost = 2 * (np + 1),
            .side = np * storage.side_x,
        };
    }
    return {
        .bottom = nx * ny,
        .side = np * (ny * storage.side_x + i64(storage.side_y) * nx),
        .xface = np * (ny + 1),
        .yface = np * nx,
        .ghost_x = np * 2 * (ny + 2),
        .ghost_y = np * 2 * nx,
    };
}
} // namespace

std::vector<ScDirBatch> sc_make_dir_batches(
    const ScGrid& grid,
    const ScDirectionSet& dirs,
    bool periodic_x,
    const std::vector<f64>& row_steps_in,
    f64 max_step_in) {
    const std::vector<f64> row_steps = row_steps_in.empty() ? sc_row_steps(grid, {}, 0) : row_steps_in;
    const f64 max_step =
        (max_step_in > 0.0) ? max_step_in : *std::max_element(row_steps.begin(), row_steps.end());
    std::vector<ScDirBatch> batches;
    const std::pair<ScRayMode, bool> groups[] = {
        {ScRayMode::Sweep, false},
        {ScRayMode::Sweep, true},
        {ScRayMode::AlongY, false},
    };
    for (const auto& [mode, flip_z] : groups) {
        std::vector<i32> group;
        std::vector<ScFrame> frames;
        for (int d = 0; d < dirs.num_dirs; ++d) {
            const ScFrame f =
                grid.dim == 3
                    ? sc_frame_3d(
                        grid,
                        dirs.mu_host(d, 0),
                        dirs.mu_host(d, 1),
                        dirs.mu_host(d, 2),
                        max_step
                    )
                    : sc_frame_2d(
                        grid,
                        dirs.mu_host(d, 0),
                        dirs.mu_host(d, 2),
                        periodic_x,
                        max_step
                    );
            if (f.mode == mode && (mode != ScRayMode::Sweep || f.flip_z == flip_z)) {
                group.push_back(d);
                frames.push_back(f);
            }
        }
        for (int start = 0; start < group.size(); start += SC_DIR_BATCH) {
            const int end = std::min(start + SC_DIR_BATCH, int(group.size()));
            ScDirBatch b;
            b.num_dirs = end - start;
            b.mode = mode;
            b.flip_z = flip_z;
            b.dir_idx_host.assign(group.begin() + start, group.begin() + end);
            b.frames_host.assign(frames.begin() + start, frames.begin() + end);
            yakl::Array<i32, 1, yakl::memHost> idx_h("sc_dir_idx", b.num_dirs);
            yakl::Array<ScFrame, 1, yakl::memHost> frames_h("sc_frames", b.num_dirs);
            for (int i = 0; i < b.num_dirs; ++i) {
                idx_h(i) = b.dir_idx_host[i];
                frames_h(i) = b.frames_host[i];
            }
            b.dir_idx = idx_h.createDeviceCopy();
            b.frames = frames_h.createDeviceCopy();
            if (mode == ScRayMode::Sweep) {
                const i32 Np = grid.num_z;
                Fp2dHost t_h("sc_row_t", b.num_dirs, Np);
                Fp2dHost ty_h;
                if (grid.dim == 3) {
                    ty_h = Fp2dHost("sc_row_ty", b.num_dirs, Np);
                }
                Fp2dHost ds_h("sc_row_ds", b.num_dirs, Np);
                Fp2dHost cell_h("sc_row_cell_ds", b.num_dirs, Np);
                for (int i = 0; i < b.num_dirs; ++i) {
                    const i32 d = b.dir_idx_host[i];
                    for (int p = 0; p < Np; ++p) {
                        // NOTE(cmo): Canonical row p is physical row Np - 1 - p
                        // when flipped, entered from the row above.
                        const f64 step = flip_z ? row_steps[Np - p] : row_steps[p];
                        sc_row_geometry(
                            grid,
                            dirs.mu_host(d, 0),
                            dirs.mu_host(d, 2),
                            step,
                            &t_h(i, p),
                            &ds_h(i, p));
                        if (grid.dim == 3) {
                            ty_h(i, p) = fp_t(
                                (std::abs(dirs.mu_host(d, 1)) * step) /
                                (std::abs(dirs.mu_host(d, 2)) * grid.dy));
                        }
                        const i32 z = flip_z ? Np - 1 - p : p;
                        cell_h(i, p) =
                            fp_t(0.5 * (row_steps[z] + row_steps[z + 1]) / std::abs(dirs.mu_host(d, 2)));
                    }
                }
                b.row_t = t_h.createDeviceCopy();
                if (grid.dim == 3) {
                    b.row_ty = ty_h.createDeviceCopy();
                }
                b.row_ds = ds_h.createDeviceCopy();
                b.row_cell_ds = cell_h.createDeviceCopy();
            }
            batches.emplace_back(std::move(b));
        }
    }
    return batches;
}

i32 sc_max_batch_dirs(const std::vector<ScDirBatch>& batches) {
    i32 result = 1;
    for (const auto& b : batches) {
        result = std::max(result, b.num_dirs);
    }
    return result;
}

/// Prepare canonical frames and characteristic geometry once, independent of wavelengths.
ScPreparedGeometry::ScPreparedGeometry(
    const ScGrid& grid_,
    const ScSlab& slab_,
    const ScDirectionSet& dirs_,
    const ScOptions& opts,
    const std::vector<f64>& z_centres,
    const std::vector<f64>& row_steps,
    f64 max_step)
    : grid(grid_), slab(slab_), dirs(dirs_), method(opts.characteristics),
      periodic_x(opts.x_boundary == ScXBoundary::Periodic),
      periodic_y(opts.y_boundary == ScXBoundary::Periodic) {
    const bool longer = opts.characteristics == ScCharacteristics::LongerCharacteristics;
    batches = sc_make_dir_batches(
        grid,
        dirs,
        periodic_x,
        row_steps,
        max_step
    );
    if (longer) {
        // Explicit coordinates may still describe a uniform grid. Only
        // roundoff-level departures from a straight spacing are accepted.
        const f64 dz = (z_centres.back() - z_centres.front()) / (z_centres.size() - 1);
        const f64 scale = std::max({std::abs(z_centres.front()), std::abs(z_centres.back()), dz});
        const f64 tolerance = 32 * std::numeric_limits<f64>::epsilon() * scale;
        bool uniform_z = true;
        for (i32 k = 0; k < z_centres.size(); ++k) {
            uniform_z = uniform_z && std::abs(z_centres[k] - (z_centres.front() + k * dz)) <= tolerance;
        }
        i32 margin_x = 0, margin_y = 0;
        for (auto& batch : batches) {
            if (batch.mode != ScRayMode::Sweep) {
                continue;
            }
            auto faces = sc_canonical_faces(z_centres, batch.flip_z);
            const i32 global_planes = z_centres.size();
            if (uniform_z) {
                for (i32 p = 0; p <= global_planes; ++p) {
                    faces[p] = p * dz;
                }
            }
            const i32 first = batch.flip_z ? global_planes - slab.z_start - grid.num_z : slab.z_start;
            auto& geometry = batch.longer;
            geometry.uniform_z = uniform_z;
            geometry.z_first = faces[first];
            geometry.dz = dz;
            if (!uniform_z) {
                yakl::Array<f64, 1, yakl::memHost> local_faces("sc_longer_z_faces", grid.num_z + 1);
                for (i32 p = 0; p <= grid.num_z; ++p) {
                    local_faces(p) = faces[first + p];
                }
                geometry.z_faces = local_faces.createDeviceCopy();
            }
            yakl::Array<ScLongerDirection, 1, yakl::memHost> directions(
                "sc_longer_directions",
                batch.num_dirs);
            for (i32 d = 0; d < batch.num_dirs; ++d) {
                const i32 dir = batch.dir_idx_host[d];
                const f64 ax = std::abs(dirs.mu_host(dir, 0));
                const f64 ay = grid.dim == 3 ? std::abs(dirs.mu_host(dir, 1)) : 0.0;
                const f64 az = std::abs(dirs.mu_host(dir, 2));
                const f64 tx = ax / (az * grid.dx);
                const f64 ty = grid.dim == 3 ? ay / (az * grid.dy) : 0.0;
                directions(d) = {
                    .slope_x = tx,
                    .slope_y = ty,
                    .inv_mu_z = 1.0 / az,
                    .delta_x = ax > 0 ? fp_t(grid.dx / ax) : FP(1e30),
                    .delta_y = ay > 0 ? fp_t(grid.dy / ay) : FP(1e30),
                    .uniform_dx = fp_t(dz * tx),
                    .uniform_dy = fp_t(dz * ty),
                    .uniform_length = fp_t(dz / az),
                };
                for (i32 p = 0; p < global_planes; ++p) {
                    const f64 x0 = faces[p] * tx, x1 = faces[p + 1] * tx;
                    const f64 y0 = faces[p] * ty, y1 = faces[p + 1] * ty;
                    const i32 nx = std::max(0, i32(std::ceil(x1) - std::floor(x0)) - 1);
                    const i32 ny = std::max(0, i32(std::ceil(y1) - std::floor(y0)) - 1);
                    if (!periodic_x) {
                        margin_x = std::max({margin_x, nx, i32(std::floor(x1) - std::floor(x0))});
                    }
                    if (!periodic_y) {
                        margin_y = std::max({margin_y, ny, i32(std::floor(y1) - std::floor(y0))});
                    }
                }
            }
            geometry.directions = directions.createDeviceCopy();
        }
        longer_layout =
            sc_longer_layout(
                grid.num_x,
                grid.num_y,
                grid.num_z,
                margin_x,
                margin_y,
                periodic_x,
                periodic_y
            );
        for (auto& batch : batches) {
            batch.longer.layout = longer_layout;
            batch.longer.uniform_z = uniform_z;
        }
    }
    // NOTE(cmo): SC: nodes per row entering through the side edge: at most the
    // columns crossed per row (+ margin); 3D: also along y.
    {
        f64 t_max = 0.0;
        f64 ty_max = 0.0;
        for (const auto& b : batches) {
            if (b.mode != ScRayMode::Sweep) {
                continue;
            }
            for (i32 i = 0; i < b.num_dirs; ++i) {
                const i32 d = b.dir_idx_host[i];
                const f64 ax = std::abs(dirs.mu_host(d, 0));
                const f64 az = std::abs(dirs.mu_host(d, 2));
                t_max = std::max(t_max, ax * max_step / (az * grid.dx));
                if (grid.dim == 3) {
                    ty_max = std::max(ty_max, std::abs(dirs.mu_host(d, 1)) * max_step / (az * grid.dy));
                }
            }
        }
        storage.side_x = std::min(grid.num_x, i32(std::floor(std::min(t_max, 1e9))) + 2);
        if (grid.dim == 3) {
            storage.side_y = std::min(grid.num_y, i32(std::floor(std::min(ty_max, 1e9))) + 2);
        }
    }
    storage.short_inflow = short_inflow_sizes(grid, storage);
    storage.longer_inflow = {
        .bottom = longer_layout.plane_size,
        .side_x = grid.num_z * (longer_layout.side_size - longer_layout.side_rows),
        .side_y = grid.num_z * longer_layout.side_rows,
    };
}

/// Match all numerical allocations so wavelength-batch selection respects the memory budget.
i64 ScPreparedGeometry::buffer_values_per_pair() const {
    const i64 plane = grid.plane_size();
    const i64 volume = plane * grid.num_z;
    const i64 medium = plane * (grid.num_z + 2 * SC_MEDIUM_HALO);
    if (method == ScCharacteristics::LongerCharacteristics) {
        return 2 * medium + 3 * volume + 2 * plane + storage.longer_inflow.total() +
               (HostDevSameSpace ? longer_layout.line_count : 0);
    }
    const i64 boundary = storage.short_inflow.total();
    return 2 * medium + 3 * volume + 2 * plane + boundary + 2 * plane;
}
