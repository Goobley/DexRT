#include <math.h>
#include <limits>
#include <magma_v2.h>
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "Populations.hpp"
#include "RadianceCascades.hpp"
#include "CrtafParser.hpp"
#include "Collisions.hpp"
#include "Voigt.hpp"
#include "StaticFormalSolution.hpp"
#include "DynamicFormalSolution.hpp"
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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;

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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
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
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
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

YAKL_INLINE void model_F_emission(const Fp3d& eta, int x, int y) {
    auto dims = eta.get_dimensions();
    vec2 c;
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
    int centre = int(dims(0) / 2);
    c(0) = centre;
    c(1) = centre;
    fp_t eta_scale = FP(20.0) / FP(60.0);
    color(0) = eta_scale;
    color(1) = FP(0.0);
    color(2) = eta_scale;
    draw_disk(eta, c, FP(30.0), color, x, y);
}

YAKL_INLINE void model_F_absorption(const Fp3d& chi, int x, int y) {
    auto dims = chi.get_dimensions();
    vec2 c;
    yakl::SArray<fp_t, 1, NUM_WAVELENGTHS> color;
    int centre = int(dims(0) / 2);
    fp_t bg = FP(1e-20);
    for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
        chi(x, y, i) = bg;
    }

    c(0) = centre;
    c(1) = centre;
    fp_t chi_scale = FP(1.0) / FP(60.0);
    color(0) = chi_scale;
    color(1) = bg;
    color(2) = chi_scale;
    draw_disk(chi, c, FP(30.0), color, x, y);

    fp_t blocker = FP(1.0);
    color(0) = blocker;
    color(1) = blocker;
    color(2) = blocker;

    int x_step = dims(0) / 12;
    int y_step = dims(1) / 12;
    for (int xx = 0; xx < 10; ++xx) {
        for (int yy = 0; yy < 10; ++yy) {
            if (xx >= 4 && xx <= 6 && yy >= 4 && yy <= 6) {
                continue;
            }
            c(0) = (xx + 1) * x_step;
            c(1) = (yy + 1) * y_step;
            draw_disk(chi, c, FP(5.0), color, x, y);
        }
    }
}

void init_state (State* state) {
    const Atmosphere& atmos = state->atmos;
    int cascade_0_x_probes, cascade_0_z_probes;
    int space_x, space_y;
    if constexpr (USE_ATMOSPHERE) {
        const auto space_dims = atmos.temperature.get_dimensions();
        space_x = space_dims(0);
        space_y = space_dims(1);
        cascade_0_x_probes = space_dims(0) / PROBE0_SPACING;
        cascade_0_z_probes = space_dims(1) / PROBE0_SPACING;
        // TODO(cmo): Need to decide whether to allow non-probe 1 spacing... it
        // probably doesn't make sense for us to have any interest in the level
        // populations not on a probe - we don't have any way to compute this outside LTE.
        state->pops = Fp3d("pops", state->atom.energy.extent(0), space_x, space_y);
        state->J = Fp3d("J", state->atom.wavelength.extent(0), space_x, space_y);
        state->J = FP(0.0);
    } else {
        space_x = MODEL_X;
        space_y = MODEL_Y;
        cascade_0_x_probes = MODEL_X / PROBE0_SPACING;
        cascade_0_z_probes = MODEL_Y / PROBE0_SPACING;
    }

    // NOTE(cmo): Allocate cascades
    if constexpr (PINGPONG_BUFFERS) {
        int64_t cascade_alloc_entries = int64_t(cascade_0_x_probes) * int64_t(cascade_0_z_probes) * PROBE0_NUM_RAYS * NUM_COMPONENTS * NUM_AZ;
        for (int l = 0; l < MAX_LEVEL + 1; ++l) {
            int64_t x_dim = cascade_0_x_probes / (1 << l);
            int64_t z_dim = cascade_0_z_probes / (1 << l);
            int64_t ray_dim = PROBE0_NUM_RAYS * (1 << (l * CASCADE_BRANCHING_FACTOR));
            int64_t cascade_size = (
                x_dim *
                z_dim *
                ray_dim *
                NUM_COMPONENTS * NUM_AZ
            );
            if (cascade_size > cascade_alloc_entries) {
                throw std::runtime_error(fmt::format(
                    "Cascade {} (with {} entries), too large for pingpong buffer ({} entries). Use a power of 2 size!",
                    l,
                    cascade_size,
                    cascade_alloc_entries
                ));
            }
            fp_t prev_radius = 0;
            if (l != 0) {
                prev_radius = PROBE0_LENGTH * (1 << ((l-1) * CASCADE_BRANCHING_FACTOR));
            }
            fp_t radius = PROBE0_LENGTH * (1 << (l * CASCADE_BRANCHING_FACTOR));
            fmt::print("[{}, {}, {}, {}->{}]\n", x_dim, z_dim, ray_dim, prev_radius, radius);
        }
        for (int i = 0; i < 2; ++i) {
            state->cascades.push_back(
                Fp5d(
                    "cascade",
                    cascade_0_x_probes,
                    cascade_0_z_probes,
                    PROBE0_NUM_RAYS,
                    NUM_COMPONENTS,
                    NUM_AZ
                )
            );
            state->cascades.back() = FP(0.0);
        }
    } else {
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
            fmt::print("[{}, {}, {}, {}->{}]\n", dims(0), dims(1), dims(2), prev_radius, radius);
            casc = FP(0.0);
        }
    }

    auto& march_state = state->raymarch_state;
    march_state.emission = Fp3d("emission_map", space_x, space_y, NUM_WAVELENGTHS);
    march_state.absorption = Fp3d("absorption_map", space_x, space_y, NUM_WAVELENGTHS);

    if constexpr (!USE_ATMOSPHERE) {
        const auto& emission = march_state.emission;
        parallel_for(
            SimpleBounds<2>(space_x, space_y),
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

        const auto& chi = march_state.absorption;
        parallel_for(
            SimpleBounds<2>(space_x, space_y),
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
        yakl::fence();
    }


    // NOTE(cmo): Allocate mipmaps if we're using them
    if constexpr (USE_MIPMAPS) {
        march_state.emission_mipmaps = std::vector<Fp3d>(MAX_LEVEL+1);
        auto& emission_mipmaps = march_state.emission_mipmaps;
        march_state.absorption_mipmaps = std::vector<Fp3d>(MAX_LEVEL+1);
        auto& absorption_mipmaps = march_state.absorption_mipmaps;
        march_state.cumulative_mipmap_factor = 0;

        auto curr_em = march_state.emission;
        auto curr_ab = march_state.absorption;

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

                if constexpr (!USE_ATMOSPHERE) {
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
                }

                emission_mipmaps[i] = new_em;
                absorption_mipmaps[i] = new_ab;

                curr_em = new_em;
                curr_ab = new_ab;
            }
        }
    }

    // NOTE(cmo): Allocate ALO array
    // TODO(cmo): this should only be when we're using the static fast path
    state->alo = Fp3d("ALO", space_x, space_y, NUM_AZ);
    const int n_level = state->atom.energy.extent(0);
    state->Gamma = Fp4d("Gamma", n_level, n_level, space_x, space_y);
}

FpConst3d final_cascade_to_J(const FpConst5d& final_cascade, const Fp3d* J_current=nullptr, int J_offset=0) {
    // [x, y, ray, wave*2, az]
    auto dims = final_cascade.get_dimensions();
    auto az_weights = get_az_weights();
    Fp3d J;
    if (J_current) {
        assert((dims(3) / 2 == 1) && "Got multiple wavelength entries when updating in existing J array, expect one at a time.");
        auto temp = J_current->slice<2>({J_offset, yakl::COLON, yakl::COLON});
        J = Fp3d("J_3d_nonown", temp.get_data(), 1, J_current->extent(1), J_current->extent(2));
        J = FP(0.0);
    } else {
        J = Fp3d("J", dims(3) / 2, dims(0), dims(1));
        J = FP(0.0);
    }
    yakl::fence();
    const fp_t angular_weight = FP(1.0) / fp_t(dims(2));
    parallel_for(
        "final_cascade_to_J",
        SimpleBounds<5>(dims(0), dims(1), dims(2), dims(3) / 2, dims(4)),
        YAKL_LAMBDA (int x, int y, int ray_idx, int la, int r) {
            fp_t ray_weight = angular_weight * az_weights(r);
            yakl::atomicAdd(J(la, x, y), ray_weight * final_cascade(x, y, ray_idx, 2 * la, r));
        }
    );
    return J;
}

void save_results(const FpConst3d& J, const FpConst3d& eta, const FpConst3d& chi, const FpConst1d& wavelengths, const FpConst3d& pops) {
    fmt::print("Saving output...\n");
    auto dims = J.get_dimensions();

    yakl::SimpleNetCDF nc;
    nc.create("output.nc", NC_CLOBBER);

    auto eta_dims = eta.get_dimensions();
    fmt::println("J: ({} {} {})", dims(0), dims(1), dims(2));
    fmt::println("eta: ({} {} {})", eta_dims(0), eta_dims(1), eta_dims(2));
    nc.write(J, "image", {"wavelength", "x", "y"});
    if (USE_ATMOSPHERE) {
        nc.write(eta, "eta", {"x", "y", "chan"});
        nc.write(chi, "chi", {"x", "y", "chan"});
        nc.write(wavelengths, "wavelength", {"wavelength"});
        nc.write(pops, "pops", {"level", "x", "y"});
    }
    nc.close();
}

int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    yakl::init(yakl::InitConfig().set_pool_initial_mb(1.55 * 1024).set_pool_grow_mb(1.55 * 1024));
    magma_init();
    {
        State state;
        magma_queue_create(0, &state.magma_queue);

        static_assert((!USE_ATMOSPHERE) || (USE_ATMOSPHERE && NUM_WAVELENGTHS == 1), "More than one wavelength per batch with USE_ATMOSPHERE - vectorisation is probably poor.");

        if constexpr (USE_ATMOSPHERE) {
            Atmosphere atmos = load_atmos("atmos.nc");
            ModelAtom<f64> model = parse_crtaf_model<f64>("../tests/test_CaII.yaml");
            CompAtom atom = to_comp_atom(model);
            state.atmos = atmos;
            state.atom = atom;
            state.phi = VoigtProfile<fp_t>(
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.15), 1024},
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(1.5e3), 64 * 1024}
            );
            state.nh_lte = HPartFn();
            fmt::println("Scale: {} m", state.atmos.voxel_scale);
        }

        // NOTE(cmo): Allocate the arrays in state, and fill emission/opacity if not using an atmosphere
        init_state(&state);
        if constexpr (USE_MIPMAPS) {
            for (int i = 0; i < MAX_LEVEL+1; ++i) {
                auto dims = state.raymarch_state.emission_mipmaps[i].get_dimensions();
                fmt::println("Cascade {} mipmap: ({}, {}, {})", i, dims(0), dims(1), dims(2));
            }
        }

        if constexpr (!USE_ATMOSPHERE) {
            for (int i = MAX_LEVEL; i >= 0; --i) {
                if constexpr (BILINEAR_FIX) {
                    compute_cascade_i_bilinear_fix_2d(&state, i);
                } else {
                    compute_cascade_i_2d(&state, i);
                }
                yakl::fence();
            }
            auto J = final_cascade_to_J(state.cascades[0]);
            FpConst3d dummy_eta, dummy_chi, dummy_pops;
            FpConst1d dummy_wave;
            save_results(J, dummy_eta, dummy_chi, dummy_wave, dummy_pops);
        } else {
            compute_lte_pops(&state);
            const bool static_soln = false;

            auto waves = state.atom.wavelength.createHostCopy();
            fp_t max_change = FP(1.0);
            if (static_soln) {
                while (max_change > FP(5e-2)) {
                    fmt::println("FS");
                    compute_collisions_to_gamma(&state);
                    yakl::fence();
                    for (int la = 0; la < waves.extent(0); ++la) {
                        // fmt::println("Computing wavelength {} ({})", la, waves(la));
                        static_formal_sol_rc(&state, la);
                        final_cascade_to_J(state.cascades[0], &state.J, la);
                        break;
                    }
                    fmt::println("Stat eq");
                    max_change = stat_eq(&state);
                }
                int la = 24;
                static_formal_sol_rc(&state, la);
                final_cascade_to_J(state.cascades[0], &state.J, la);
            } else {
                for (int la = 0; la < waves.extent(0); ++la) {
                    fmt::println("Computing wavelength {} ({})", la, waves(la));
                    dynamic_formal_sol_rc(&state, la);
                    final_cascade_to_J(state.cascades[0], &state.J, la);
                }
                int la = 24;
                dynamic_formal_sol_rc(&state, la);
                final_cascade_to_J(state.cascades[0], &state.J, la);
            }
            save_results(
                state.J,
                state.raymarch_state.emission,
                state.raymarch_state.absorption,
                state.atom.wavelength,
                state.pops
            );
        }
        magma_queue_destroy(state.magma_queue);
    }
    magma_finalize();
    yakl::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
