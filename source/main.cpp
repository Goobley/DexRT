#include <math.h>
#include <limits>
#include "Config.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "Populations.hpp"
#include "RayMarching.hpp"
#include "RadianceIntervals.hpp"
#include "RadianceCascades.hpp"
#include "CrtafParser.hpp"
#ifdef HAVE_MPI
    #include "YAKL_pnetcdf.h"
#else
    #include "YAKL_netcdf.h"
#endif
#include <vector>
#include <string>
#include <optional>
#include <fmt/core.h>

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

    auto dims = emission.get_dimensions();

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

    centre(0) = dims(0) / 2;
    centre(1) = dims(1) / 2;
    color(0) = FP(1.0);
    color(1) = FP(1.0);
    color(2) = FP(1.0);
    draw_disk(emission, centre, 2, color, x, y);
}

YAKL_INLINE void model_B(Fp3d emission, int x, int y) {
    auto dims = emission.get_dimensions();
    int centre = int(dims(0) / 2);
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
    auto dims = emission.get_dimensions();
    int centre = int(dims(0) / 2);
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

    c(0) = dims(0) - 200;
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
    auto dims = chi.get_dimensions();
    int centre = int(dims(0) / 2);
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

    c(0) = dims(0) - 200;
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

YAKL_INLINE void model_E_emission(const Fp3d& emission, int x, int y) {
    auto dims = emission.get_dimensions();
    int centre = int(dims(0) / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    c(0) = centre;
    c(1) = centre;
    fp_t emission_scale = FP(10.0) / FP(80.0);
    // fp_t emission_scale = FP(40.0);
    color(0) = emission_scale;
    color(1) = FP(0.0);
    color(2) = emission_scale;
    draw_disk(emission, c, 40, color, x, y);
}

YAKL_INLINE void model_E_absorption(const Fp3d& chi, int x, int y) {
    auto dims = chi.get_dimensions();
    int centre = int(dims(0) / 2);
    vec2 c;
    yakl::SArray<fp_t, 1, 3> color;
    // fp_t bg = FP(1e-10);
    fp_t bg = FP(1e-20);
    for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
        chi(x, y, i) = bg;
    }

    c(0) = centre;
    c(1) = centre;
    fp_t chi_scale = FP(1.0) / FP(80.0);
    // fp_t chi_scale = FP(4.0);
    color(0) = chi_scale;
    color(1) = bg;
    color(2) = chi_scale;
    draw_disk(chi, c, 40, color, x, y);
}

void init_state (State* state, const Atmosphere& atmos) {
    const auto space_dims = atmos.temperature.get_dimensions();
    const int cascade_0_x_probes = space_dims(0) / PROBE0_SPACING;
    const int cascade_0_z_probes = space_dims(1) / PROBE0_SPACING;
    
    for (int l = 0; l < MAX_LEVEL + 1; ++l) {
        state->cascades.push_back(
            Fp5d(
                "cascade",
                cascade_0_x_probes / (1 << l),
                cascade_0_z_probes / (1 << l),
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

    auto& march_state = state->raymarch_state;
    march_state.emission = Fp3d("emission_map", space_dims(0), space_dims(1), NUM_WAVELENGTHS);

    {
        const auto& emission = march_state.emission;
        parallel_for(
            SimpleBounds<2>(space_dims(0), space_dims(1)),
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

    march_state.absorption = Fp3d("absorption_map", space_dims(0), space_dims(1), NUM_WAVELENGTHS);
    {
        const auto& chi = march_state.absorption;
        parallel_for(
            SimpleBounds<2>(space_dims(0), space_dims(1)),
            YAKL_LAMBDA (int x, int y) {
#ifndef ABSORPTION_MODEL
                LIGHT_MODEL(chi, x, y);
                for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                    if (chi(x, y, i) == FP(0.0)) {
                        chi(x, y, i) = FP(1e-10);
                    } else if (chi(x, y, i) == FP(1e-6)) {
                        chi(x, y, i) = FP(3.0)true;
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

    yakl::fence();

    if (USE_MIPMAPS) {
        march_state.emission_mipmaps = std::vector<Fp3d>(MAX_LEVEL+1);
        auto& emission_mipmaps = march_state.emission_mipmaps;
        march_state.absorption_mipmaps = std::vector<Fp3d>(MAX_LEVEL+1);
        auto& absorption_mipmaps = march_state.absorption_mipmaps;
        march_state.cumulative_mipmap_factor = 0;

        auto& curr_em = march_state.emission;
        auto& curr_ab = march_state.absorption;

        for (int i = 0; i <= MAX_LEVEL; ++i) {
            if (MIPMAP_FACTORS[i] == 0) {
                emission_mipmaps[i] = curr_em;
                absorption_mipmaps[i] = curr_ab;
                if (i == 0) {
                    march_state.cumulative_mipmap_factor(i) = 0;
                } else {
                    march_state.cumulative_mipmap_factor(i) = march_state.cumulative_mipmap_factor(i-1);
                }
            } else {
                int factor = MIPMAP_FACTORS[i];
                if (i == 0) {
                    march_state.cumulative_mipmap_factor(i) = factor;
                } else {
                    march_state.cumulative_mipmap_factor(i) = march_state.cumulative_mipmap_factor(i-1) + factor;
                }

                auto dims = curr_em.get_dimensions();
                Fp3d new_em = Fp3d("emission_mipmap", dims(0) / (1 << factor), dims(1) / (1 << factor), dims(2));
                Fp3d new_ab = Fp3d("absorption_mipmap", dims(0) / (1 << factor), dims(1) / (1 << factor), dims(2));
                auto new_dims = new_em.get_dimensions();

                // NOTE(cmo): Very basic averaging approach
                parallel_for(
                    SimpleBounds<2>(new_dims(0), new_dims(1)),
                    YAKL_LAMBDA (int x, int y) {
                        mipmap_arr(curr_em, new_em, factor, x, y);
                    }
                );
                parallel_for(
                    SimpleBounds<2>(new_dims(0), new_dims(1)),
                    YAKL_LAMBDA (int x, int y) {
                        mipmap_arr(curr_ab, new_ab, factor, x, y);
                    }
                );
                yakl::fence();

                emission_mipmaps[i] = new_em;
                absorption_mipmaps[i] = new_ab;

                curr_em = new_em;
                curr_ab = new_ab;
            }
        }
    }
}

void save_results(const FpConst5d& final_cascade) {
    fmt::print("Saving output...\n");
    auto dims = final_cascade.get_dimensions();
    const yakl::Array<double, 4, yakl::memDevice> fp64_copy("fp64_copy", dims(0), dims(1), dims(2), NUM_WAVELENGTHS);
    auto az_weights = get_az_weights();
    parallel_for(
        SimpleBounds<3>(dims(0), dims(1), dims(2)),
        YAKL_LAMBDA (int x, int y, int ray_idx) {
            for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
                fp64_copy(x, y, ray_idx, i) = 0.0;
                for (int r = 0; r < NUM_AZ; ++r) {
                    fp64_copy(x, y, ray_idx, i)  += az_weights(r) * final_cascade(x, y, ray_idx, 2*i, r);
                }
            }
        }
    );
    yakl::fence();
    const auto out_dims = fp64_copy.get_dimensions();

#ifdef HAVE_MPI
    yakl::SimplePNetCDF nc;
    nc.create("output.nc");
    std::vector<MPI_Offset> starts(4, 0);
    nc.create_dim("x", out_dims(0));
    nc.create_dim("y", out_dims(1));
    nc.create_dim("ray", out_dims(2));
    nc.create_dim("col", out_dims(3));
    nc.create_var<decltype(fp64_copy)::type>("image", {"x", "y", "ray", "col"});
    nc.enddef();
    nc.write_all(fp64_copy, "image", starts);
    nc.close();
#else
    yakl::SimpleNetCDF nc;
    nc.create("output.nc", NC_CLOBBER);
    nc.write(fp64_copy, "image", {"x", "y", "ray", "col"});
    nc.close();
#endif
}

int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    yakl::init();
    {
        Atmosphere atmos = load_atmos("atmos.nc");
        State state;
        init_state(&state, atmos);
        if (USE_MIPMAPS) {
            for (int i = 0; i < MAX_LEVEL+1; ++i) {
                auto dims = state.raymarch_state.emission_mipmaps[i].get_dimensions();
                fmt::println("Cascade {} mipmap: ({}, {}, {})", i, dims(0), dims(1), dims(2));
            }
        }
        auto dims = state.cascades[0].get_dimensions();

        for (int i = MAX_LEVEL; i >= 0; --i) {
            if (BILINEAR_FIX) {
                compute_cascade_i_bilinear_fix_2d(&state, i);
            } else {
                compute_cascade_i_2d(&state, i);
            }
            yakl::fence();
        }

        dims = state.cascades[0].get_dimensions();
        save_results(state.cascades[0]);
    }
    yakl::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
