#include <math.h>
#include <limits>
#include "YAKL.h"
#include "Config.hpp"
#include "Types.hpp"
#include <stdio.h>
#ifdef HAVE_MPI
#include "YAKL_pnetcdf.h"
#endif
#include <vector>

using yakl::c::parallel_for;
using yakl::c::Bounds;
using yakl::c::SimpleBounds;

YAKL_INLINE void model_original(Fp3d emission, int x, int y) {
    constexpr fp_t intensity_scale = FP(3.0);
    if (x >= 25 && x < 35) {
        if (y >= 235 && y < 275) {
            emission(x, y, 0) = intensity_scale;
        }
    }

    if (x >= 295 && x < 311) {
        if (y >= 145 && y < 365) {
            emission(x, y, 0) = FP(1e-6);
            emission(x, y, 1) = FP(1e-6);
            emission(x, y, 2) = FP(1e-6);
        }
    }

    if (x >= 253 && x < 260) {
        if (y >= 253 && y < 260) {
            emission(x, y, 1) = FP(4.0) * intensity_scale;
        }
    }

    if (x >= 220 && x < 222) {
        if (y >= 220 && y < 222) {
            emission(x, y, 0) = FP(1.414) * intensity_scale;
            emission(x, y, 2) = FP(1.414) * intensity_scale;
        }
    }

    if (x >= 405 && x < 410) {
        if (y >= 252 && y < 262) {
            emission(x, y, 2) = intensity_scale;
        }
    }
}

YAKL_INLINE void draw_disk(
    Fp3d emission,
    vec2 centre, 
    fp_t radius, 
    yakl::SArray<fp_t, 1, NUM_COMPONENTS-1> color,
    int x,
    int y
) {
    fp_t dx = fp_t(x) - centre(0);
    fp_t dy = fp_t(y) - centre(1);

    if ((dx * dx + dy * dy) <= (radius * radius)) {
        for (int i = 0; i < NUM_COMPONENTS-1; ++i) {
            emission(x, y, i) = color(i);
        }
    }
}

YAKL_INLINE void model_A(Fp3d emission, int x, int y) {
    vec2 centre;
    yakl::SArray<fp_t, 1, 3> color;

    centre(0) = 30;
    centre(1) = 30;
    color(0) = FP(10.0);
    color(1) = FP(10.0);
    color(2) = FP(10.0);
    draw_disk(emission, centre, 30, color, x, y);

    centre(0) = 50;
    centre(1) = 180;
    color(0) = FP(1e-6);
    color(1) = FP(1e-6);
    color(2) = FP(1e-6);
    draw_disk(emission, centre, 6, color, x, y);

    centre(0) = CANVAS_X / 2;
    centre(1) = CANVAS_Y / 2;
    color(0) = FP(1.0);
    color(1) = FP(1.0);
    color(2) = FP(1.0);
    draw_disk(emission, centre, 2, color, x, y);
}

YAKL_INLINE void model_B(Fp3d emission, int x, int y) {
    int centre = int(CANVAS_X / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    color(0) = FP(0.0);
    color(1) = FP(1.0);
    color(2) = FP(0.0);
    for (int cx = centre - 256; cx < centre + 256; ++cx) {
        c(0) = centre;
        c(1) = cx;

        draw_disk(emission, c, 6, color, x, y);
    }

    color(0) = FP(1e-6);
    color(1) = FP(1e-6);
    color(2) = FP(1e-6);
    for (int cx = centre - 50; cx < centre + 50; ++cx) {
        c(0) = centre + 100;
        c(1) = cx;

        draw_disk(emission, c, 6, color, x, y);
    }

    color(0) = FP(10.0);
    color(1) = FP(0.0);
    color(2) = FP(0.0);
    for (int cx = centre - 50; cx < centre + 50; ++cx) {
        c(0) = 100;
        c(1) = cx;

        draw_disk(emission, c, 6, color, x, y);
    }

    c(0) = centre + 400;
    c(1) = centre;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(0.5);
    draw_disk(emission, c, 40, color, x, y);
}

void init_state (State* state) {
    for (int l = 0; l < MAX_LEVEL + 1; ++l) {
        state->cascades.push_back(
            Fp4d(
                "cascade",
                PROBES_IN_CASCADE_0 / (1 << l),
                PROBES_IN_CASCADE_0 / (1 << l),
                PROBE0_NUM_RAYS * (1 << (l * CASCADE_BRANCHING_FACTOR)),
                NUM_COMPONENTS
            )
        );

        auto casc = state->cascades[l];
        auto dims = casc.get_dimensions();
        printf("[%d, %d, %d, %d]\n", dims(0), dims(1), dims(2), dims(3));
        parallel_for(
            SimpleBounds<3>(dims(0), dims(1), dims(2)),
            YAKL_LAMBDA (int i, int j, int k) {
                for (int m = 0; m < NUM_COMPONENTS; ++m) {
                    casc(i, j, k, m) = FP(0.0);
                }
            }
        );
    }

    state->emission = Fp3d("emission_map", CANVAS_X, CANVAS_Y, NUM_COMPONENTS - 1);

    {
        auto emission = state->emission;
        fp_t intensity_scale = FP(3.0);
        parallel_for(
            SimpleBounds<2>(CANVAS_X, CANVAS_Y),
            YAKL_LAMBDA (int x, int y) {
                emission(x, y, 0) = FP(0.0);
                emission(x, y, 1) = FP(0.0);
                emission(x, y, 2) = FP(0.0);

                // model_original(emission, x, y);
                model_A(emission, x, y);
                // model_B(emission, x, y);
            }
        );
    }
    yakl::fence();
}

YAKL_INLINE yakl::SArray<fp_t, 1, NUM_COMPONENTS> raymarch(const Fp3d& domain, vec2 ray_start, vec2 direction, fp_t distance) {
    int steps = distance * yakl::max(yakl::abs(direction(0)), yakl::abs(direction(1)));
    vec2 step;
    step(0) = distance * direction(0) / steps;
    step(1) = distance * direction(1) / steps;

    vec2 pos;
    pos(0) = ray_start(0);
    pos(1) = ray_start(1);

    ivec2 sample_coord;
    yakl::SArray<fp_t, 1, NUM_COMPONENTS-1> sample;
    yakl::SArray<fp_t, 1, NUM_COMPONENTS> result;
    for (int i = 0; i < steps; ++i) {
        sample_coord(0) = int(std::round(pos(0)));
        sample_coord(1) = int(std::round(pos(1)));

        if (sample_coord(0) >= CANVAS_X || sample_coord(1) >= CANVAS_Y || sample_coord(0) < 0 || sample_coord(1) < 0) {
            for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
                result(i) = FP(0.0);
            }
            result(NUM_COMPONENTS-1) = FP(1.0);
            return result;
        }

        for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
            sample(i) = domain(sample_coord(0), sample_coord(1), i);
        }

        bool hits = false;
        for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
            hits |= bool(sample(i) != FP(0.0));
            if (hits) {
                break;
            }
        }

        if (hits) {
            for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
                result(i) = sample(i);
            }
            result(NUM_COMPONENTS - 1) = FP(0.0);
            return result;
        }
        
        pos(0) += step(0);
        pos(1) += step(1);
    }

    for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
        result(i) = FP(0.0);
    }
    result(NUM_COMPONENTS-1) = FP(1.0);
    return result;
}

YAKL_INLINE yakl::SArray<fp_t, 1, NUM_COMPONENTS> merge_intervals(yakl::SArray<fp_t, 1, NUM_COMPONENTS> closer, yakl::SArray<fp_t, 1, NUM_COMPONENTS> further) {
    fp_t transmission = closer(NUM_COMPONENTS-1);
    if (transmission == FP(0.0)) {
        return closer;
    }

    for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
        closer(i) += transmission * further(i);
    }
    closer(NUM_COMPONENTS-1) = closer(NUM_COMPONENTS-1) * further(NUM_COMPONENTS-1);
    return closer;
}

void compute_cascade_i (
    State* state,
    int cascade_idx
) {
    auto& emission = state->emission;
    printf("Cascade idx: %d, vec size: %d\n", cascade_idx, state->cascades.size());
    auto& cascade_i = state->cascades[cascade_idx];
    auto cascade_ip = cascade_i;
    if (cascade_idx != MAX_LEVEL) {
        cascade_ip = state->cascades[cascade_idx + 1];
    }
    auto dims = cascade_i.get_dimensions();
    auto edims = emission.get_dimensions();
    auto upper_dims = cascade_ip.get_dimensions();
    printf("[%d, %d, %d, %d]\n", dims(0), dims(1), dims(2), dims(3));
    printf("[%d, %d, %d]\n", edims(0), edims(1), edims(2));
    parallel_for(
        SimpleBounds<3>(dims(2), dims(0), dims(1)),
        YAKL_LAMBDA (int ray_idx, int u, int v) {
            // NOTE(cmo): u, v probe indices
            // NOTE(cmo): x, y world coords
            for (int i = 0; i < NUM_COMPONENTS; ++i) {
                cascade_i(u, v, ray_idx, i) = FP(0.0);
            }

            int upper_cascade_idx = cascade_idx + 1;
            fp_t spacing = PROBE0_SPACING * (1 << cascade_idx);
            fp_t upper_spacing = PROBE0_SPACING * (1 << upper_cascade_idx);
            int num_rays = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            int upper_num_rays = PROBE0_NUM_RAYS * (1 << (upper_cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t radius = PROBE0_LENGTH * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t prev_radius = FP(0.0);
            if (cascade_idx != 0) {
                prev_radius = PROBE0_LENGTH * (1 << ((cascade_idx-1) * CASCADE_BRANCHING_FACTOR));
            }

            int cx = int((u + FP(0.5)) * spacing);
            int cy = int((v + FP(0.5)) * spacing);
            fp_t distance = radius - prev_radius;

            // NOTE(cmo): bc: bottom corner of 2x2 used for bilinear interp on next cascade
            // uc: upper corner
            int upper_u_bc = int((u - 1) / 2);
            int upper_v_bc = int((v - 1) / 2);
            fp_t u_bc_weight = FP(0.25) + FP(0.5) * (u % 2);
            fp_t v_bc_weight = FP(0.25) + FP(0.5) * (v % 2);
            fp_t u_uc_weight = FP(1.0) - u_bc_weight;
            fp_t v_uc_weight = FP(1.0) - v_bc_weight;
            int u_bc = yakl::max(upper_u_bc, 0);
            int v_bc = yakl::max(upper_v_bc, 0);
            int u_uc = yakl::min(u_bc+1, (int)upper_dims(0));
            int v_uc = yakl::min(v_bc+1, (int)upper_dims(1));

            fp_t angle = FP(2.0) * FP(M_PI) / num_rays * (ray_idx + FP(0.5));
            vec2 direction;
            direction(0) = yakl::cos(angle);
            direction(1) = yakl::sin(angle);
            vec2 start;
            start(0) = cx + prev_radius * direction(0);
            start(1) = cy + prev_radius * direction(1);

            int upper_ray_start_idx = ray_idx * (1 << CASCADE_BRANCHING_FACTOR);
            int num_rays_per_ray = 1 << CASCADE_BRANCHING_FACTOR;
            fp_t ray_weight = FP(1.0) / num_rays_per_ray;

            if (BILINEAR_FIX) {
            } else {
                auto sample = raymarch(emission, start, direction, distance);
                decltype(sample) upper_sample(FP(0.0));
                // NOTE(cmo): Sample upper cascade.
                if (cascade_idx != MAX_LEVEL && sample(NUM_COMPONENTS-1) > FP(0.0)) {
                    for (
                        int upper_ray_idx = upper_ray_start_idx;
                        upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                        ++upper_ray_idx
                    ) {
                        for (int i = 0; i < NUM_COMPONENTS; ++i) {
                            fp_t u_11 = cascade_ip(u_bc, v_bc, upper_ray_idx, i);
                            fp_t u_21 = cascade_ip(u_uc, v_bc, upper_ray_idx, i);
                            fp_t u_12 = cascade_ip(u_bc, v_uc, upper_ray_idx, i);
                            fp_t u_22 = cascade_ip(u_uc, v_uc, upper_ray_idx, i);
                            fp_t term = (
                                v_bc_weight * (u_bc_weight * u_11 + u_uc_weight * u_21) +
                                v_uc_weight * (u_bc_weight * u_12 + u_uc_weight * u_22)
                            );
                            upper_sample(i) += ray_weight * term;
                            if (u == 50 && v == 120) {
                                printf("<[%f, %f, %f, %f]>\n", upper_sample(i));
                            }
                        }
                    }
                }
                auto merged = merge_intervals(sample, upper_sample);
                for (int i = 0; i < NUM_COMPONENTS; ++i) {
                    cascade_i(u, v, ray_idx, i) = merged(i);
                }
            }
        }
    );
}

#ifdef HAVE_MPI
void save_results(Fp4d final_cascade) {
    printf("Writing\n");
    auto dims = final_cascade.get_dimensions();
    yakl::Array<double, 4, yakl::memDevice> fp64_copy("fp64_copy", dims(0), dims(1), dims(2), dims(3) - 1);
    parallel_for(
        SimpleBounds<3>(dims(0), dims(1), dims(2)),
        YAKL_LAMBDA (int x, int y, int ray_idx) {
            for (int i = 0; i < NUM_COMPONENTS - 1; ++i) {
                fp64_copy(x, y, ray_idx, i)  = final_cascade(x, y, ray_idx, i);
            }
        }
    );
    yakl::fence();
    auto host_copy = fp64_copy.createHostCopy();


    //Error reporting routine for the PNetCDF I/O
    auto ncwrap = []( int ierr , int line ) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error at line: %d\n", line);
            printf("%s\n", ncmpi_strerror(ierr));
            exit(-1);
        }
    };

    auto out_dims = host_copy.get_dimensions();
    int ncid, x_dimid, y_dimid, ray_dimid, col_dimid, im_id;
    ncwrap(ncmpi_create(MPI_COMM_WORLD, "output.nc", NC_CLOBBER, MPI_INFO_NULL, &ncid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "x", out_dims(0), &x_dimid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "y", out_dims(1), &y_dimid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "ray", out_dims(2), &ray_dimid), __LINE__);
    ncwrap(ncmpi_def_dim(ncid, "col", out_dims(3), &col_dimid), __LINE__);
    int dimids[4];
    dimids[0] = x_dimid;
    dimids[1] = y_dimid;
    dimids[2] = ray_dimid;
    dimids[3] = col_dimid;

    ncwrap(ncmpi_def_var(ncid, "image", NC_DOUBLE, 4, dimids, &im_id), __LINE__);
    ncwrap(ncmpi_enddef(ncid), __LINE__);
    MPI_Offset start[4] = {0};
    MPI_Offset count[4];
    count[0] = out_dims(0); 
    count[1] = out_dims(1); 
    count[2] = out_dims(2);
    count[3] = out_dims(3);
    ncwrap(ncmpi_put_vara_double_all(ncid, im_id, start, count, host_copy.data()), __LINE__);
    ncwrap(ncmpi_close(ncid), __LINE__);
}
#endif

int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    yakl::init();
    {
        State state;
        init_state(&state);
        auto dims = state.cascades[0].get_dimensions();
        printf("[[%d, %d, %d, %d]]\n", dims(0), dims(1), dims(2), dims(3));


        for (int i = MAX_LEVEL; i >= 0; --i) {
            compute_cascade_i(&state, i);
            yakl::fence();
        }

#ifdef HAVE_MPI
        dims = state.cascades[0].get_dimensions();
        save_results(state.cascades[0]);
#endif
    }
    yakl::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
