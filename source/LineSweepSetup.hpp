#if !defined(DEXRT_LINE_SWEEP_SETUP_HPP)
#define DEXRT_LINE_SWEEP_SETUP_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "BlockMap.hpp"
#include <vector>

/// A single line to be swept
struct LsLine {
    vec2 o;
    fp_t t0 = FP(0.0);
    fp_t t1 = FP(1e24);
    fp_t first_sample = FP(0.0);
    i32 num_samples = 0;

    LsLine() = default;
    KOKKOS_INLINE_FUNCTION LsLine(vec2 o_, fp_t t0_=FP(0.0), fp_t t1_=FP(1e24)) :
        o(o_),
        t0(t0_),
        t1(t1_)
    {}

    KOKKOS_INLINE_FUNCTION IntersectionResult intersects(const GridBbox& bbox, const vec2& d, fp_t step) const {
        fp_t t0_ = t0;
        fp_t t1_ = t1;

        for (int ax = 0; ax < 2; ++ax) {
            fp_t a = fp_t(bbox.min(ax));
            fp_t b = fp_t(bbox.max(ax));
            if (a >= b) {
                return IntersectionResult{
                    .intersects = false,
                    .t0 = t0_,
                    .t1 = t1_
                };
            }

            a = (a - o(ax)) / d(ax);
            b = (b - o(ax)) / d(ax);
            if (a > b) {
                fp_t temp = b;
                b = a;
                a = temp;
            }

            if (a > t0_) {
                t0_ = a;
            }
            if (b < t1_) {
                t1_ = b;
            }
            if (t0_ > t1_) {
                return IntersectionResult{
                    .intersects = false,
                    .t0 = t0_,
                    .t1 = t1_
                };
            }
        }
        return IntersectionResult{
            .intersects = true,
            .t0 = t0_,
            .t1 = t1_
        };
    }

    KOKKOS_INLINE_FUNCTION bool clip(const GridBbox& bbox, const vec2& d, fp_t step) {
        IntersectionResult result = intersects(bbox, d, step);
        if (!result.intersects) {
            return false;
        }

        t0 = result.t0;
        t1 = result.t1;
        first_sample = int(t0) + FP(0.5) * step;
        if (first_sample < t0) {
            first_sample += step;
        }
        num_samples = int((t1 - first_sample) / step) + 1;
        return true;
    }
};

/// The metadata for a set of lines for one direction
struct LineSetDescriptor {
    vec2 origin;
    vec2 d;
    fp_t step;
    i32 primary_axis;
    i32 line_start_idx; /// the index of the first line in this set
    i32 num_lines;
    i32 total_steps;
};

/// The metatdata for set of lines for a set of directions
struct DirSetDescriptor {
    fp_t step;
    i32 total_lines;
    i32 total_steps;
};

/// Contains the descriptors and informations for a set of lines used on a cascade (or a cascade subset if using DIR_BY_DIR)
struct CascadeLineSet {
    DirSetDescriptor dir_set_desc;
    yakl::Array<LineSetDescriptor, 1> line_set_desc; // meta data for each lineset
    yakl::Array<LsLine, 1> lines;
    yakl::Array<i32, 1> line_storage_start_idx; /// starting index of storage associated with a line
};

struct LineSweepStorage {
    Fp1d source_term;
    Fp1d transmittance;
};

struct LineSweepData {
    std::vector<CascadeLineSet> cascade_sets;
    LineSweepStorage storage;
};

LineSweepData construct_line_sweep_data(const State& state, int max_cascade);

#else
#endif