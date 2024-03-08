#include <math.h>
#include <limits>
#include "Config.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include "RayMarching.hpp"
#include "RadianceIntervals.hpp"
#ifdef HAVE_MPI
    #include "YAKL_pnetcdf.h"
#endif
#include <vector>
#include <string>
#include <optional>
#include <fmt/core.h>

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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color,
    int x,
    int y
) {
    fp_t dx = fp_t(x) - centre(0);
    fp_t dy = fp_t(y) - centre(1);

    if ((dx * dx + dy * dy) <= (radius * radius)) {
        for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
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

YAKL_INLINE void model_D_emission(Fp3d emission, int x, int y) {
    int centre = int(CANVAS_X / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    c(0) = centre + 400;
    c(1) = centre;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(2.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = centre - 400;
    c(1) = centre;
    color(0) = FP(2.0);
    color(1) = FP(0.0);
    color(2) = FP(0.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre - 400;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(0.5);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre + 400;
    color(0) = FP(0.5);
    color(1) = FP(0.0);
    color(2) = FP(0.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = 200;
    c(1) = 200;
    color(0) = FP(0.0);
    color(1) = FP(3.0);
    color(2) = FP(0.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = CANVAS_X - 200;
    c(1) = 200;
    color(0) = FP(0.0);
    color(1) = FP(0.0);
    color(2) = FP(3.0);
    draw_disk(emission, c, 40, color, x, y);

    c(0) = 400;
    c(1) = 700;
    color(0) = FP(3.0);
    color(1) = FP(0.0);
    color(2) = FP(3.0);
    draw_disk(emission, c, 40, color, x, y);
}

YAKL_INLINE void model_D_absorption(Fp3d chi, int x, int y) {
    int centre = int(CANVAS_X / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    fp_t bg = FP(1e-10);
    for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
        chi(x, y, i) = bg;
    }
    c(0) = centre + 400;
    c(1) = centre;
    color(0) = bg;
    color(1) = bg;
    color(2) = FP(0.5);
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre - 400;
    c(1) = centre;
    color(0) = FP(0.5);
    color(1) = bg;
    color(2) = bg;
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre - 400;
    color(0) = bg;
    color(1) = bg;
    color(2) = FP(0.5);
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre;
    c(1) = centre + 400;
    color(0) = FP(0.5);
    color(1) = bg;
    color(2) = bg;
    draw_disk(chi, c, 40, color, x, y);

    c(0) = centre + 340;
    c(1) = centre;
    color(0) = FP(0.2);
    color(1) = FP(0.2);
    color(2) = FP(0.2);
    draw_disk(chi, c, 6, color, x, y);
    c(0) = centre - 340;
    c(1) = centre;
    draw_disk(chi, c, 6, color, x, y);

    int box_size = 250;
    if (x >= centre - box_size && x < centre + box_size && y >= centre - box_size && y < centre + box_size) {
        fp_t chi_r = FP(1e-4);
        chi(x, y, 0) = chi_r;
        chi(x, y, 1) = FP(1e2) * chi_r;
        chi(x, y, 2) = FP(1e2) * chi_r;
    }

    c(0) = 200;
    c(1) = 200;
    color(0) = bg;
    color(1) = FP(1.0);
    color(2) = bg;
    draw_disk(chi, c, 40, color, x, y);

    c(0) = CANVAS_X - 200;
    c(1) = 200;
    color(0) = bg;
    color(1) = bg;
    color(2) = FP(1.0);
    draw_disk(chi, c, 40, color, x, y);

    c(0) = 400;
    c(1) = 700;
    color(0) = FP(1.0);
    color(1) = bg;
    color(2) = FP(1.0);
    draw_disk(chi, c, 40, color, x, y);
}

yakl::SArray<fp_t, 1, NUM_AZ> get_az_rays() {
    yakl::SArray<fp_t, 1, NUM_AZ> az_rays;
    for (int r = 0; r < NUM_AZ; ++r) {
        az_rays(r) = INCL_RAYS[r];
    }
    return az_rays;
}

yakl::SArray<fp_t, 1, NUM_AZ> get_az_weights() {
    yakl::SArray<fp_t, 1, NUM_AZ> az_weights;
    for (int r = 0; r < NUM_AZ; ++r) {
        az_weights(r) = AZ_WEIGHTS[r];
    }
    return az_weights;
}

void init_state (State* state) {
    for (int l = 0; l < MAX_LEVEL + 1; ++l) {
        state->cascades.push_back(
            Fp5d(
                "cascade",
                PROBES_IN_CASCADE_0 / (1 << l),
                PROBES_IN_CASCADE_0 / (1 << l),
                PROBE0_NUM_RAYS * (1 << (l * CASCADE_BRANCHING_FACTOR)),
                NUM_COMPONENTS,
                NUM_AZ
            )
        );

        auto casc = state->cascades[l];
        auto dims = casc.get_dimensions();
        fp_t prev_radius = 0;
        if (l != 0) {
            prev_radius = PROBE0_LENGTH * (1 << ((l-1) * CASCADE_BRANCHING_FACTOR));
        }
        fp_t radius = PROBE0_LENGTH * (1 << (l * CASCADE_BRANCHING_FACTOR));
        fmt::print("[{}, {}, {}, {}, {}->{}]\n", dims(0), dims(1), dims(2), dims(3), prev_radius, radius);
        parallel_for(
            SimpleBounds<3>(dims(0), dims(1), dims(2)),
            YAKL_LAMBDA (int i, int j, int k) {
                for (int m = 0; m < NUM_COMPONENTS; ++m) {
                    for (int p = 0; p < NUM_AZ; ++p) {
                        casc(i, j, k, m, p) = FP(0.0);
                    }
                }
            }
        );
    }

    state->emission = Fp3d("emission_map", CANVAS_X, CANVAS_Y, NUM_WAVELENGTHS);

    {
        auto emission = state->emission;
        parallel_for(
            SimpleBounds<2>(CANVAS_X, CANVAS_Y),
            YAKL_LAMBDA (int x, int y) {
                for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                    emission(x, y, i) = FP(0.0);
                }

                // model_original(emission, x, y);
                // model_A(emission, x, y);
                // model_B(emission, x, y);
                LIGHT_MODEL(emission, x, y);
            }
        );
    }

#ifndef TRACE_OPAQUE_LIGHTS
    state->absorption = Fp3d("absorption_map", CANVAS_X, CANVAS_Y, NUM_WAVELENGTHS);
    {
        auto chi = state->absorption;
        parallel_for(
            SimpleBounds<2>(CANVAS_X, CANVAS_Y),
            YAKL_LAMBDA (int x, int y) {
#ifndef ABSORPTION_MODEL
                LIGHT_MODEL(chi, x, y);
                for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                    if (chi(x, y, i) == FP(0.0)) {
                        chi(x, y, i) = FP(1e-10);
                    } else if (chi(x, y, i) == FP(1e-6)) {
                        chi(x, y, i) = FP(3.0);
                    } else {
                        chi(x, y, i) /= FP(2.0);
                    }
                }
#else
                ABSORPTION_MODEL(chi, x, y);
#endif
            }
        );

    }
#endif

    yakl::fence();
}

void compute_cascade_i (
    State* state,
    int cascade_idx
) {
    const FpConst3d emission = state->emission;
    const FpConst3d chi = state->absorption;
    const auto& cascade_i = state->cascades[cascade_idx];
    FpConst5d cascade_ip = cascade_i;
    if (cascade_idx != MAX_LEVEL) {
        cascade_ip = state->cascades[cascade_idx + 1];
    }
    auto dims = cascade_i.get_dimensions();
    auto edims = emission.get_dimensions();
    auto upper_dims = cascade_ip.get_dimensions();

    auto az_rays = get_az_rays();

    std::string cascade_name = fmt::format("Cascade {}", cascade_idx);
    yakl::timer_start(cascade_name.c_str());
    parallel_for(
        SimpleBounds<3>(dims(2), dims(0), dims(1)),
        YAKL_LAMBDA (int ray_idx, int u, int v) {
            // NOTE(cmo): u, v probe indices
            // NOTE(cmo): x, y world coords
            for (int i = 0; i < NUM_COMPONENTS; ++i) {
                for (int j = 0; j < NUM_AZ; ++j) {
                    cascade_i(u, v, ray_idx, i, j) = FP(0.0);
                }
            }

            int upper_cascade_idx = cascade_idx + 1;
            fp_t spacing = PROBE0_SPACING * (1 << cascade_idx);
            fp_t upper_spacing = PROBE0_SPACING * (1 << upper_cascade_idx);
            int num_rays = PROBE0_NUM_RAYS * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            int upper_num_rays = PROBE0_NUM_RAYS * (1 << (upper_cascade_idx * CASCADE_BRANCHING_FACTOR));
            fp_t radius = PROBE0_LENGTH * (1 << (cascade_idx * CASCADE_BRANCHING_FACTOR));
            if (LAST_CASCADE_TO_INFTY && cascade_idx == MAX_LEVEL) {
                radius = LAST_CASCADE_MAX_DIST;
            }
            fp_t prev_radius = FP(0.0);
            if (cascade_idx != 0) {
                prev_radius = PROBE0_LENGTH * (1 << ((cascade_idx-1) * CASCADE_BRANCHING_FACTOR));
            }

            fp_t cx = (u + FP(0.5)) * spacing;
            fp_t cy = (v + FP(0.5)) * spacing;
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
                auto sample = raymarch(emission, chi, start, direction, distance, az_rays);
                decltype(sample) upper_sample(FP(0.0));
                // NOTE(cmo): Sample upper cascade.
                // if (cascade_idx != MAX_LEVEL && sample(NUM_COMPONENTS-1) > FP(0.0)) {
                if (cascade_idx != MAX_LEVEL) {
                    for (
                        int upper_ray_idx = upper_ray_start_idx;
                        upper_ray_idx < upper_ray_start_idx + num_rays_per_ray;
                        ++upper_ray_idx
                    ) {
                        for (int i = 0; i < NUM_COMPONENTS; ++i) {
                            for (int r = 0; r < NUM_AZ; ++r) {
                                if (az_rays(r) == FP(0.0)) {
                                    // NOTE(cmo): Can't merge the in-out of page ray.
                                    continue;
                                }
                                fp_t u_11 = cascade_ip(u_bc, v_bc, upper_ray_idx, i, r);
                                fp_t u_21 = cascade_ip(u_uc, v_bc, upper_ray_idx, i, r);
                                fp_t u_12 = cascade_ip(u_bc, v_uc, upper_ray_idx, i, r);
                                fp_t u_22 = cascade_ip(u_uc, v_uc, upper_ray_idx, i, r);
                                fp_t term = (
                                    v_bc_weight * (u_bc_weight * u_11 + u_uc_weight * u_21) +
                                    v_uc_weight * (u_bc_weight * u_12 + u_uc_weight * u_22)
                                );
                                upper_sample(i, r) += ray_weight * term;
                            }
                        }
                    }
                }
                auto merged = merge_intervals(sample, upper_sample, az_rays);
                for (int i = 0; i < NUM_COMPONENTS; ++i) {
                    for (int r = 0; r < NUM_AZ; ++r) {
                        cascade_i(u, v, ray_idx, i, r) = merged(i, r);
                    }
                }
            }
        }
    );
    yakl::timer_stop(cascade_name.c_str());
}

#ifdef HAVE_MPI
void save_results(Fp5d final_cascade) {
    printf("Writing\n");
    auto dims = final_cascade.get_dimensions();
    yakl::Array<double, 4, yakl::memDevice> fp64_copy("fp64_copy", dims(0), dims(1), dims(2), NUM_WAVELENGTHS);
    auto az_weights = get_az_weights();
    parallel_for(
        SimpleBounds<3>(dims(0), dims(1), dims(2)),
        YAKL_LAMBDA (int x, int y, int ray_idx) {
            for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
#ifdef TRACE_OPAQUE_LIGHTS
                fp64_copy(x, y, ray_idx, i)  = final_cascade(x, y, ray_idx, i);
#else
                fp64_copy(x, y, ray_idx, i) = 0.0;
                for (int r = 0; r < NUM_AZ; ++r) {
                    fp64_copy(x, y, ray_idx, i)  += az_weights(r) * final_cascade(x, y, ray_idx, 2*i, r);
                }
#endif
            }
        }
    );
    yakl::fence();
    auto host_copy = fp64_copy.createHostCopy();


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
