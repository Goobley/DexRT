#include "LineSweepSetup.hpp"
#include "RcUtilsModes.hpp"
#include "BlockMap.hpp"
#include "CascadeState.hpp"

int LineSweepData::get_cascade_subset_idx(int cascade_idx, int subset_idx) const {
    constexpr int RcFlags = RC_flags_storage();
    constexpr int num_subsets = subset_tasks_per_cascade<RcFlags>();
    return (cascade_idx - LINE_SWEEP_START_CASCADE) * num_subsets + subset_idx;
}

KOKKOS_INLINE_FUNCTION vec2 select_origin(const vec2& dir, const GridBbox& bbox) {
    vec2 result;
    result(0) = bbox.min(0);
    result(1) = bbox.min(1);
    if (dir(1) >= FP(0.0)) {
        if (dir(0) < FP(0.0)) {
            result(0) = bbox.max(0);
        }
    } else {
        if (dir(0) >= FP(0.0)) {
            result(1) = bbox.max(1);
        } else {
            result(0) = bbox.max(0);
            result(1) = bbox.max(1);
        }
    }
    return result;
}

CascadeLineSet construct_line_sweep_subset(const State& state, int cascade_idx, int subset_idx) {
    constexpr int RcMode = RC_flags_storage();
    CascadeRays ray_set = cascade_compute_size<RcMode>(state.c0_size, cascade_idx);
    const fp_t step = probe_spacing(cascade_idx);
    const auto& bbox = state.mr_block_map.block_map.bbox;

    int num_lines = 0;
    int num_steps = 0;
    i32 max_steps = 0;
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    std::vector<LsLine> subset_lines;
    std::vector<LineSetDescriptor> subset_line_set_desc;
    std::vector<i32> line_storage_start_idx;
    line_storage_start_idx.emplace_back(0);
    for (
        int phi_idx = ray_subset.start_flat_dirs;
        phi_idx < ray_subset.start_flat_dirs + ray_subset.num_flat_dirs;
        ++phi_idx
    ) {
        // NOTE(cmo): Line sweeping does its directional inversion here to match
        // the raymarching later on.
        vec2 dir = ray_dir(ray_set, phi_idx);
        dir(0) = -dir(0);
        dir(1) = -dir(1);

        vec2 origin = select_origin(dir, bbox);
        bool step_along_x = std::abs(dir(1) / dir(0)) >= FP(1.0);
        int primary_axis = step_along_x ? 0 : 1;
        vec2 primary_vec(FP(0.0));
        primary_vec(primary_axis) = FP(1.0);

        vec2 normal0 = primary_vec;
        vec2 normal_mul;
        normal_mul(0) = FP(1.0);
        normal_mul(1) = FP(-1.0);

        origin = origin + FP(0.5) * primary_vec * std::copysign(FP(1.0), dir(primary_axis));
        LsLine base_line(origin);
        bool intersect = base_line.clip(bbox, dir, step);
        KOKKOS_ASSERT(intersect);
        subset_lines.emplace_back(base_line);
        i32 line_set_start_idx = num_lines;
        i32 line_set_start_steps = num_steps;
        num_lines += 1;
        num_steps += base_line.num_samples;
        max_steps = std::max(max_steps, base_line.num_samples);
        for (int ni = 0; ni < 2; ++ni) {
            vec2 normal = normal_mul(ni) * normal0;
            vec2 pos(origin);
            while (true) {
                pos = pos + normal * step;
                LsLine line(pos);
                if (!line.clip(bbox, dir, step)) {
                    break;
                }
                if (line.t1 - line.t0 < FP(0.5) * step) {
                    continue;
                }
                num_steps += line.num_samples;
                max_steps = std::max(max_steps, line.num_samples);
                num_lines += 1;
                subset_lines.emplace_back(line);
            }
        }
        std::sort(
            std::begin(subset_lines) + line_set_start_idx,
            std::end(subset_lines),
            [=](const LsLine& a, const LsLine& b) {
                return a.o(primary_axis) < b.o(primary_axis);
            }
        );
        subset_line_set_desc.emplace_back(LineSetDescriptor{
            .origin = origin,
            .d = dir,
            .step = step,
            .primary_axis = primary_axis,
            .line_start_idx = line_set_start_idx,
            .num_lines = num_lines - line_set_start_idx,
            .total_steps = num_steps - line_set_start_steps
        });
        line_storage_start_idx.resize(num_lines, 0);
        for (int i = std::max(line_set_start_idx, 1); i < num_lines; ++i) {
            line_storage_start_idx[i] = line_storage_start_idx[i-1] + subset_lines[i-1].num_samples;
        }
    }

    i32 total_steps = 0;
    for (const auto& ls_desc : subset_line_set_desc) {
        total_steps += ls_desc.total_steps;
    }
    std::vector<i32> ls_start_idx(subset_line_set_desc.size());
    for (int i = 0; i < ls_start_idx.size(); ++i) {
        ls_start_idx[i] = subset_line_set_desc[i].line_start_idx;
    }


    DirSetDescriptor dir_set_desc{
        .step = step,
        .total_lines = i32(subset_lines.size()),
        .total_steps = total_steps,
        .max_line_steps = max_steps
    };
    auto line_set_desc_d = yakl::Array<LineSetDescriptor, 1, yakl::memHost>("line_set_desc", subset_line_set_desc.data(), subset_line_set_desc.size()).createDeviceCopy();
    auto lines_d = yakl::Array<LsLine, 1, yakl::memHost>("lines", subset_lines.data(), subset_lines.size()).createDeviceCopy();
    auto line_storage_start_idx_d = yakl::Array<i32, 1, yakl::memHost>("line_start_idx", line_storage_start_idx.data(), line_storage_start_idx.size()).createDeviceCopy();
    auto line_set_start_idx_d = yakl::Array<i32, 1, yakl::memHost>("line_set_start_idx", ls_start_idx.data(), ls_start_idx.size()).createDeviceCopy();
    CascadeLineSet line_data{
        .dir_set_desc = dir_set_desc,
        .line_set_desc = line_set_desc_d,
        .lines = lines_d,
        .line_storage_start_idx = line_storage_start_idx_d,
        .line_set_start_idx = line_set_start_idx_d
    };

    return line_data;
}

LineSweepData construct_line_sweep_data(const State& state, int max_cascade) {

    int min_cascade = LINE_SWEEP_START_CASCADE;
    constexpr int RcMode = RC_flags_storage();
    constexpr int num_subsets = subset_tasks_per_cascade<RcMode>();

    std::vector<CascadeLineSet> cascade_sets;
    for (int cascade_idx = min_cascade; cascade_idx <= max_cascade; ++cascade_idx) {
        for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
            cascade_sets.emplace_back(construct_line_sweep_subset(state, cascade_idx, subset_idx));
        }
    }

    i32 max_entries = 0;
    for (const auto& cs : cascade_sets) {
        max_entries = std::max(max_entries, cs.dir_set_desc.total_steps);
    }
    state.println("Allocating 2x{} kB for line-sweeping storage", max_entries * sizeof(fp_t) * state.c0_size.num_incl / 1024);

    // NOTE(cmo): This doesn't include space for each wavelength.
    LineSweepStorage storage {
        .source_term = Fp2d("ls_source", max_entries, state.c0_size.num_incl),
        .transmittance = Fp2d("ls_trans", max_entries, state.c0_size.num_incl)
    };

    return LineSweepData{
        .cascade_sets = cascade_sets,
        .storage = storage
    };
}
