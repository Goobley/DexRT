#include <math.h>
#include <limits>
#include <magma_v2.h>
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include "Populations.hpp"
#include "RadianceCascades2.hpp"
#include "CrtafParser.hpp"
#include "Collisions.hpp"
#include "Voigt.hpp"
#include "StaticFormalSolution.hpp"
#include "DynamicFormalSolution.hpp"
#include "GammaMatrix.hpp"
#include "PromweaverBoundary.hpp"
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
        cascade_0_x_probes = space_dims(1) / PROBE0_SPACING;
        cascade_0_z_probes = space_dims(0) / PROBE0_SPACING;
        // TODO(cmo): Need to decide whether to allow non-probe 1 spacing... it
        // probably doesn't make sense for us to have any interest in the level
        // populations not on a probe - we don't have any way to compute this outside LTE.
        const int n_level_total = state->adata.energy.extent(0);
        state->pops = Fp3d("pops", n_level_total, space_x, space_y);
        state->J = Fp3d("J", state->adata.wavelength.extent(0), space_x, space_y);
        state->J = FP(0.0);

        // state->Gamma = Fp4d("Gamma", n_level, n_level, space_x, space_y);
        for (int ia = 0; ia < state->adata_host.num_level.extent(0); ++ia) {
            const int n_level = state->adata_host.num_level(ia);
            state->Gamma.emplace_back(
                Fp4d("Gamma", n_level, n_level, space_x, space_y)
            );
        }

        state->pw_bc = load_bc(ATMOS_PATH, state->adata.wavelength);
        if constexpr (USE_BC) {
            state->boundary = BoundaryType::Promweaver;
        } else {
            state->boundary = BoundaryType::Zero;
        }
    } else {
        space_x = MODEL_X;
        space_y = MODEL_Y;
        cascade_0_x_probes = MODEL_X / PROBE0_SPACING;
        cascade_0_z_probes = MODEL_Y / PROBE0_SPACING;
        state->boundary = BoundaryType::Zero;
    }

    CascadeRays c0_rays;
    c0_rays.num_probes(0) = cascade_0_x_probes;
    c0_rays.num_probes(1) = cascade_0_z_probes;
    c0_rays.num_flat_dirs = PROBE0_NUM_RAYS;
    c0_rays.num_incl = NUM_INCL;
    c0_rays.wave_batch = WAVE_BATCH;

    state->c0_size = cascade_rays_to_storage<PREAVERAGE>(c0_rays);
    if constexpr (USE_ATMOSPHERE) {
        state->alo = Fp5d(
            "ALO",
            state->c0_size.num_probes(1),
            state->c0_size.num_probes(0),
            state->c0_size.num_flat_dirs,
            state->c0_size.wave_batch,
            state->c0_size.num_incl
        );
        state->dynamic_opac = decltype(state->dynamic_opac)(
            "Dynamic Emis/Opac",
            state->c0_size.num_probes(1),
            state->c0_size.num_probes(0),
            state->c0_size.wave_batch
        );
    }


    Fp1dHost muy("muy", NUM_INCL);
    Fp1dHost wmuy("wmuy", NUM_INCL);
    for (int i = 0; i < NUM_INCL; ++i) {
        muy(i) = INCL_RAYS[i];
        wmuy(i) = INCL_WEIGHTS[i];
    }
    state->incl_quad.muy = muy.createDeviceCopy();
    state->incl_quad.wmuy = wmuy.createDeviceCopy();
}

FpConst3d final_cascade_to_J(
    const FpConst1d& final_cascade,
    const CascadeStorage& c0_dims,
    const Fp3d& J,
    const InclQuadrature incl_quad,
    int la_start,
    int la_end
) {
    const fp_t phi_weight = FP(1.0) / fp_t(c0_dims.num_flat_dirs);
    int wave_batch = la_end - la_start;

    parallel_for(
        "final_cascade_to_J",
        SimpleBounds<5>(
            c0_dims.num_probes(1),
            c0_dims.num_probes(0),
            c0_dims.num_flat_dirs,
            wave_batch,
            c0_dims.num_incl),
        YAKL_LAMBDA (int z, int x, int phi_idx, int wave, int theta_idx) {
            fp_t ray_weight = phi_weight * incl_quad.wmuy(theta_idx);
            int la = la_start + wave;
            ivec2 coord;
            coord(0) = x;
            coord(1) = z;
            ProbeIndex idx{
                .coord=coord,
                .dir=phi_idx,
                .incl=theta_idx,
                .wave=wave
            };
            const fp_t sample = probe_fetch(final_cascade, c0_dims, idx);
            yakl::atomicAdd(J(la, z, x), ray_weight * sample);
        }
    );
    return J;
}

void save_results(const FpConst3d& J, const FpConst3d& eta, const FpConst3d& chi, const FpConst1d& wavelengths, const FpConst3d& pops, const FpConst1d& casc=FpConst1d(), const FpConst5d& alo=FpConst5d()) {
    fmt::print("Saving output...\n");
    auto dims = J.get_dimensions();

    yakl::SimpleNetCDF nc;
    nc.create("output.nc", yakl::NETCDF_MODE_REPLACE);

    auto eta_dims = eta.get_dimensions();
    fmt::println("J: ({} {} {})", dims(0), dims(1), dims(2));
    fmt::println("eta: ({} {} {})", eta_dims(0), eta_dims(1), eta_dims(2));
    nc.write(J, "image", {"wavelength", "z", "x"});
    if (USE_ATMOSPHERE) {
        nc.write(eta, "eta", {"z", "x", "wave_batch"});
        nc.write(chi, "chi", {"z", "x", "wave_batch"});
        nc.write(wavelengths, "wavelength", {"wavelength"});
        nc.write(pops, "pops", {"level", "z", "x"});
        if (casc.initialized()) {
            nc.write(casc, "cascade", {"cascade_shape"});
        }
        if (alo.initialized()) {
            nc.write(alo, "alo", {"z", "x", "dir", "wave_batch", "incl"});
        }
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
            Atmosphere atmos = load_atmos(ATMOS_PATH);
            ModelAtom<f64> model = parse_crtaf_model<f64>("../tests/test_CaII.yaml");
            AtomicDataHostDevice<fp_t> atomic_data = to_atomic_data<fp_t, f64>({model});
            state.adata = atomic_data.device;
            state.adata_host = atomic_data.host;
            state.have_h = atomic_data.have_h_model;
            state.atoms = extract_atoms(atomic_data.device, atomic_data.host);
            GammaAtomsAndMapping gamma_atoms = extract_atoms_with_gamma_and_mapping(atomic_data.device, atomic_data.host);
            state.atoms_with_gamma = gamma_atoms.atoms;
            state.atoms_with_gamma_mapping = gamma_atoms.mapping;
            state.atmos = atmos;
            state.phi = VoigtProfile<fp_t>(
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.15), 1024},
                VoigtProfile<fp_t>::Linspace{FP(0.0), FP(1.5e3), 64 * 1024}
            );
            state.nh_lte = HPartFn();
            fmt::println("Scale: {} m", state.atmos.voxel_scale);
        }

        // NOTE(cmo): Allocate the arrays in state, and fill emission/opacity if not using an atmosphere
        init_state(&state);
        CascadeState casc_state = CascadeState_new(state.c0_size, MAX_CASCADE);

        // if constexpr (!USE_ATMOSPHERE) {
        //     for (int i = MAX_LEVEL; i >= 0; --i) {
        //         if constexpr (BILINEAR_FIX) {
        //             compute_cascade_i_bilinear_fix_2d(&state, i);
        //         } else {
        //             compute_cascade_i_2d(&state, i);
        //         }
        //         yakl::fence();
        //     }
        //     auto J = final_cascade_to_J(state.cascades[0]);
        //     FpConst3d dummy_eta, dummy_chi, dummy_pops;
        //     FpConst1d dummy_wave;
        //     save_results(J, dummy_eta, dummy_chi, dummy_wave, dummy_pops);
        // } else {
            compute_lte_pops(&state);
            constexpr bool non_lte = false;
            constexpr bool static_soln = true;
            auto& waves = state.adata_host.wavelength;
            auto fs_fn = dynamic_formal_sol_rc;
            if (static_soln) {
                fs_fn = static_formal_sol_rc;
            }
            fp_t max_change = FP(1.0);
            if (non_lte) {
                constexpr int max_iters = 300;
                int i = 0;
                while (max_change > FP(1e-3) && i < max_iters) {
                    fmt::println("FS {}", i);
                    compute_collisions_to_gamma(&state);
                    state.J = FP(0.0);
                    yakl::fence();
                    for (
                        int la_start = 0;
                        la_start < waves.extent(0);
                        la_start += state.c0_size.wave_batch
                    ) {
                        int la_end = std::min(la_start + state.c0_size.wave_batch, int(waves.extent(0)));

                        fs_fn(
                            state,
                            casc_state,
                            la_start,
                            la_end
                        );
                        final_cascade_to_J(
                            casc_state.i_cascades[0],
                            state.c0_size,
                            state.J,
                            state.incl_quad,
                            la_start,
                            la_end
                        );
                    }
                    // NOTE(cmo): Fixup now done in stateq
                    // fixup_gamma(flat_Gamma);
                    yakl::fence();
                    fmt::println("Stat eq");
                    max_change = stat_eq<f64>(&state);
                    i += 1;
                }
            } else {
                state.J = FP(0.0);
                yakl::fence();
                for (int la_start = 0; la_start < waves.extent(0); la_start += state.c0_size.wave_batch) {
                    int la_end = std::min(la_start + state.c0_size.wave_batch, int(waves.extent(0)));
                    fmt::println(
                        "Computing wavelengths [{}, {}] ({}, {}) (static: {})",
                        la_start,
                        la_end,
                        waves(la_start),
                        waves(la_end-1),
                        static_soln
                    );
                    fs_fn(state, casc_state, la_start, la_end);
                    final_cascade_to_J(
                        casc_state.i_cascades[0],
                        state.c0_size,
                        state.J,
                        state.incl_quad,
                        la_start,
                        la_end
                    );
                }
            }
            save_results(
                state.J,
                casc_state.eta,
                casc_state.chi,
                state.adata.wavelength,
                state.pops,
                casc_state.i_cascades[casc_state.i_cascades.size() - 1],
                state.alo
            );
        // }
        magma_queue_destroy(state.magma_queue);
    }
    magma_finalize();
    yakl::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
