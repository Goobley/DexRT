#include "ScRadiation.hpp"
#include "ScGammaAccumulate.hpp"
#include "ScFormalSolver.hpp"
#include "LoopUtils.hpp"
#include "Atmosphere.hpp"
#include <type_traits>

void sc_accumulate_gamma(
    const ScState& state,
    const ScMediumCoefficients& medium_coefficients,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    i32 la_start,
    i32 wave_batch,
    bool lambda_iterate,
    const Fp2d& J,
    const Rad16Array& previous_J
) {
    JasUnpack(state, background);
    JasUnpack(medium, chi);
    JasUnpack(state, phi, pops, adata, wphi);
    JasUnpack(medium, S);
    JasUnpack(solution, I_ext);
    const bool use_background = background.enabled;
    const auto flat_atmos = flatten<const fp_t>(state.atmos);
    JasUnpack(batch, frames, dir_idx);
    JasUnpack(dirs, mu, wmu);
    JasUnpack(state, cells);
    JasUnpack(cells, runs);
    const i32 num_dirs = batch.num_dirs;

    Fp1dHost wl_weights_h("sc_wl_weights", wave_batch);
    for (int wave = 0; wave < wave_batch; ++wave) {
        wl_weights_h(wave) = gamma_wavelength_weight(state.adata_host, la_start + wave);
    }
    const Fp1d wl_weights = wl_weights_h.createDeviceCopy();

    const auto& lte_scratch = medium_coefficients.n_star;
    const GammaAccumArrayState out = gamma_accum_array_state(state.precon_R, &state.rate_diag);
    // NOTE(cmo): J (the batch's slot, if given) from the same reads.
    const bool accumulate_J = J.initialized();
    const i64 J_offset = state.cells.owned_offset();
    constexpr i32 halo = SC_MEDIUM_HALO;

    constexpr i32 G = SC_CELL_GROUP;
    static_assert(BLOCK_SIZE * BLOCK_SIZE % G == 0);
    dex_parallel_for(
        "SC compute Gamma",
        FlatLoop<1>(state.cells.num_runs),
        YAKL_LAMBDA(i64 r) {
            JasUse(G, halo);
            const auto piece = runs(r);
            const i64 ks0 = cells.ks(piece.x0, piece.y, piece.z);
            const i32 n = piece.length();
            AtmosPointParams local_atmos[G];
            fp_t vel[G][3];
            for (int c = 0; c < n; ++c) {
                const i64 ks = ks0 + c;
                local_atmos[c].temperature = flat_atmos.temperature(ks);
                local_atmos[c].ne = flat_atmos.ne(ks);
                local_atmos[c].vturb = flat_atmos.vturb(ks);
                local_atmos[c].nhtot = flat_atmos.nh_tot(ks);
                local_atmos[c].nh0 = flat_atmos.nh0(ks);
                local_atmos[c].vel = FP(0.0);
                vel[c][0] = flat_atmos.vx(ks);
                vel[c][1] = flat_atmos.vy(ks);
                vel[c][2] = flat_atmos.vz(ks);
            }

            for (int wave = 0; wave < wave_batch; ++wave) {
                const int la = la_start + wave;
                const auto active_set = slice_active_set(adata, la);
                const auto active_cont_set = slice_active_cont_set(adata, la);
                const auto make_cell = [&](int c) {
                    return ScGammaCell{
                        .adata = adata,
                        .phi = phi,
                        .pops = pops,
                        .lte_scratch = lte_scratch,
                        .wphi = wphi,
                        .out = out,
                        .active_set = active_set,
                        .active_cont_set = active_cont_set,
                        .atmos = local_atmos[c],
                        .ks = ks0 + c,
                        .la = la,
                        .wl_weight = wl_weights(wave),
                        .line_cache = &medium_coefficients,
                        .wave = wave,
                    };
                };
                GammaStaticSums statics[G];
                GammaRayAngularIntegral moments[G];
                fp_t J_entry[G];
                for (int c = 0; c < n; ++c) {
                    ScGammaCell cell = make_cell(c);
                    fp_t eta_bg = FP(0.0);
                    fp_t chi_bg = FP(0.0);
                    if (use_background) {
                        const i64 ks = ks0 + c;
                        const BackgroundTerms bg = background_terms(
                            background,
                            la,
                            ks,
                            local_atmos[c].temperature,
                            local_atmos[c].ne,
                            local_atmos[c].nhtot
                        );
                        eta_bg =
                            bg.chi_abs * planck_wavelength(adata.wavelength(la), local_atmos[c].temperature) +
                            bg.sca * previous_J(wave, ks);
                        chi_bg = bg.chi_abs + bg.sca;
                    }
                    sc_gamma_static_terms(&cell, eta_bg, chi_bg);
                    statics[c] = cell.statics;
                    J_entry[c] = FP(0.0);
                }

                const ScGammaCell cell0 = make_cell(0);
                for (int d = 0; d < num_dirs; ++d) {
                    const ScFrame f = frames(d);
                    const i32 dir = dir_idx(d);
                    ScGammaRays<G> rays;
                    rays.weight = wmu(dir);
                    const fp_t mu_x = mu(dir, 0);
                    const fp_t mu_y = mu(dir, 1);
                    const fp_t mu_z = mu(dir, 2);
                    // NOTE(cmo): Raw row pointers, and the direction's x
                    // flip hoisted out of the loop over the group, so that it
                    // vectorises with unit stride.
                    const i64 idx0 = sc_canonical_idx(f, piece.x0, piece.y, piece.z);
                    const i64 mat_offset = i64(halo) * sc_plane_size(f);
                    const fp_t* __restrict I_nl_p = &I_ext(d, wave, idx0);
                    const fp_t* __restrict chi_p = &chi(d, wave, idx0 + mat_offset);
                    const fp_t* __restrict S_p = &S(d, wave, idx0 + mat_offset);
                    const fp_t* __restrict b_p = &solution.alo(d, wave, idx0);
                    const fp_t* __restrict q_p = &solution.one_minus_alo(d, wave, idx0);
                    if constexpr (G > 1) {
                        // NOTE(cmo): The next direction's rows are cold lines
                        // (one per array per direction and wavelength).
                        if (d + 1 < num_dirs) {
                            const ScFrame fn = frames(d + 1);
                            const i64 idx_n =
                                sc_canonical_idx(fn, piece.x0 + (fn.flip_x ? n - 1 : 0), piece.y, piece.z);
                            const i64 mat_n = idx_n + i64(halo) * sc_plane_size(fn);
                            // NOTE(cmo): Both ends of the group (two cache lines).
                            const auto row = [&](const fp_t* p) {
                                sc_prefetch(p);
                                sc_prefetch(p + (n - 1));
                            };
                            row(&I_ext(d + 1, wave, idx_n));
                            row(&chi(d + 1, wave, mat_n));
                            row(&S(d + 1, wave, mat_n));
                            row(&solution.alo(d + 1, wave, idx_n));
                            row(&solution.one_minus_alo(d + 1, wave, idx_n));
                        }
                    }
                    const auto ray_inputs = [&](auto step) {
                        constexpr i64 st = decltype(step)::value;
                        SC_GROUP_SIMD
                        for (int c = 0; c < n; ++c) {
                            const i64 i = st * c;
                            const fp_t alo = b_p[i];
                            const fp_t chi0 = chi_p[i];
                            const fp_t S0 = S_p[i];
                            const fp_t I_nl_c = I_nl_p[i];
                            const fp_t I = I_nl_c + alo * S0;
                            J_entry[c] += rays.weight * (accumulate_J ? I : FP(0.0));
                            rays.vel[c] = vel[c][0] * mu_x + vel[c][1] * mu_y + vel[c][2] * mu_z;
                            // NOTE(cmo): Lambda iteration: no preconditioning, so the
                            // split form with I_ext = I, q = 1, psi* = 0.
                            rays.I_ext[c] = lambda_iterate ? I : I_nl_c;
                            rays.one_minus_alo[c] = lambda_iterate ? FP(1.0) : q_p[i];
                        }
                    };
                    if (f.flip_x) {
                        ray_inputs(std::integral_constant<i64, -1>{});
                    } else {
                        ray_inputs(std::integral_constant<i64, 1>{});
                    }
                    sc_accumulate_gamma_rays<G>(
                        cell0,
                        local_atmos,
                        statics,
                        rays,
                        moments,
                        n
                    );
                }
                for (int c = 0; c < n; ++c) {
                    ScGammaCell cell = make_cell(c);
                    accumulate_gamma_continua(
                        sc_gamma_cell(cell, cell.ks, statics[c]),
                        cell.atmos,
                        moments[c]
                    );
                    if (accumulate_J) {
                        J(wave, ks0 + c - J_offset) += J_entry[c];
                    }
                }
            }
        });
    Kokkos::fence();
}

template <bool Dense>
void sc_accumulate_J(
    const ScState& state,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    const Fp2d& J,
    i32 wave_batch
) {
    JasUnpack(medium, S);
    constexpr i32 halo = SC_MEDIUM_HALO;
    JasUnpack(batch, frames, dir_idx);
    JasUnpack(dirs, wmu);
    const i32 num_dirs = batch.num_dirs;
    JasUnpack(state, cells);
    const auto list = cells.list();
    const i64 J_offset = cells.owned_offset();

    dex_parallel_for(
        Dense ? "SC accumulate J (dense)" : "SC accumulate J",
        FlatLoop<2>(wave_batch, Dense ? cells.num_owned_cells() : list.num()),
        YAKL_LAMBDA(i32 wave, i64 i) {
            JasUse(halo);
            // NOTE(cmo): k indexes the owned rows (J).
            const i64 k = Dense ? i : list(i) - J_offset;
            const Coord3 coord = cells.coord(k + J_offset);
            fp_t entry = FP(0.0);
            for (int d = 0; d < num_dirs; ++d) {
                const ScFrame f = frames(d);
                const i64 idx = sc_canonical_idx(f, coord.x, coord.y, coord.z);
                const fp_t I = solution.intensity(d, wave, idx, S(d, wave, idx + halo * sc_plane_size(f)));
                entry += wmu(dir_idx(d)) * I;
            }
            J(wave, k) += entry;
        });
    Kokkos::fence();
}
template void sc_accumulate_J<false>(
    const ScState&,
    const ScDirectionSet&,
    const ScDirBatch&,
    const ScMedium&,
    const ScSolution&,
    const Fp2d&,
    i32
);
template void sc_accumulate_J<true>(
    const ScState&,
    const ScDirectionSet&,
    const ScDirBatch&,
    const ScMedium&,
    const ScSolution&,
    const Fp2d&,
    i32
);

void sc_store_emergent(
    const ScGrid& grid,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    const Fp3d& outgoing,
    const Fp3d& emergent,
    i32 la_start,
    i32 wave_batch
) {
    JasUnpack(medium, S);
    JasUnpack(batch, frames, dir_idx);
    JasUnpack(dirs, mu);
    const i32 top = grid.num_z - 1;
    dex_parallel_for(
        "SC store emergent",
        FlatLoop<3>(batch.num_dirs, wave_batch, grid.plane_size()),
        YAKL_LAMBDA(i32 d, i32 wave, i64 xy) {
            const i32 dir = dir_idx(d);
            if (mu(dir, 2) <= FP(0.0)) {
                return;
            }
            const ScFrame f = frames(d);
            const i32 y = xy / f.num_s;
            const i32 x = xy - i64(y) * f.num_s;
            if (f.mode == ScRayMode::Sweep) {
                const i32 s = f.flip_x ? f.num_s - 1 - x : x;
                const i32 sy = f.flip_y ? f.num_sy - 1 - y : y;
                emergent(la_start + wave, dir, xy) = outgoing(d, wave, i64(sy) * f.num_s + s);
                return;
            }
            const i64 idx = sc_canonical_idx(f, x, y, top);
            emergent(la_start + wave, dir, xy) =
                solution.intensity(d, wave, idx, S(d, wave, idx + i64(SC_MEDIUM_HALO) * sc_plane_size(f)));
        });
    Kokkos::fence();
}
