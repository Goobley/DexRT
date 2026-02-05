#include "FewFreqFormalSolution.hpp"
#include "RadianceCascades.hpp"
#include "Populations.hpp"
#include "EmisOpac.hpp"
#include "LteHPops.hpp"
#include "GammaMatrix.hpp"
#include "Atmosphere.hpp"
#include "RcUtilsModes.hpp"
#include "MergeToJ.hpp"

FewFreqSetup compute_wavelength_setup(const State& state) {
    JasUnpack(state, adata_host);
    // NOTE(cmo): Build wavelength grid as per few_freq config
    std::vector<i32> wave_idx;
    wave_idx.reserve(128);
    std::vector<fp_t> wavelength;
    wavelength.reserve(128);
    std::vector<fp_t> bandwidth;
    bandwidth.reserve(128);
    std::vector<FewFreqSetup::TransType> trans_type;
    trans_type.reserve(128);
    std::vector<i32> trans_idx;
    trans_idx.reserve(128);

    for (int kr = 0; kr < adata_host.lines.extent(0); ++kr) {
        wavelength.emplace_back(adata_host.lines(kr).lambda0);
        bandwidth.emplace_back(adata_host.lines(kr).bandwidth);
        const i32 la = std::lower_bound(
            adata_host.wavelength.begin(),
            adata_host.wavelength.end(),
            adata_host.lines(kr).lambda0
        ) - adata_host.wavelength.begin();
        wave_idx.emplace_back(la);
        trans_type.emplace_back(FewFreqSetup::TransType::Line);
        trans_idx.emplace_back(kr);
    }
    if (state.config.few_freq.cont_mode == FewFrequencyContMode::Hydrogenic) {
        throw std::runtime_error("Hydrogenic continua not implemented yet");
    }
    for (int kr = 0; kr < adata_host.continua.extent(0); ++kr) {
        const auto& cont = adata_host.continua(kr);
        // NOTE(cmo): This is a right-sided (red-sided) integral. It probably
        // overestimates a little, but shouldn't be too serious
        fp_t prev_wl = adata_host.wavelength(cont.blue_idx);
        for (int la = cont.blue_idx; la < cont.red_idx; ++la) {
            if (adata_host.wavelength(la) - prev_wl >= state.config.few_freq.cont_sample_nm || la == cont.red_idx - 1) {
                const fp_t delta = adata_host.wavelength(la) - prev_wl;
                bandwidth.emplace_back(delta);
                wavelength.emplace_back(adata_host.wavelength(la));
                wave_idx.emplace_back(la);
                trans_type.emplace_back(FewFreqSetup::TransType::Continuum);
                trans_idx.emplace_back(kr);

                prev_wl = adata_host.wavelength(la);
            }
        }
    }

    return FewFreqSetup {
        .wave_idx = yakl::Array<i32, 1, yakl::memHost>("wave_idx", wave_idx.data(), wave_idx.size()).createDeviceCopy(),
        .wavelength = Fp1d("wavelength", wavelength.data(), wavelength.size()).createDeviceCopy(),
        .bandwidth = Fp1d("bandwidth", bandwidth.data(), bandwidth.size()).createDeviceCopy(),
        .trans_type = yakl::Array<FewFreqSetup::TransType, 1, yakl::memHost>("trans_type", trans_type.data(), trans_type.size()).createDeviceCopy(),
        .trans_idx = yakl::Array<i32, 1, yakl::memHost>("trans_idx", trans_idx.data(), trans_idx.size()).createDeviceCopy()
    };
}

void few_freq_compute_gamma(
    const State& state,
    const CascadeState& casc_state,
    const FewFreqSetup& fs_setup,
    const Fp2d& n_star,
    const CascadeCalcSubset& subset
) {
    JasUnpack(subset, la_start, la_end, subset_idx);
    JasUnpack(state, pops, adata, mr_block_map);
    using namespace ConstantsFP;
    const auto flatmos = flatten<const fp_t>(state.atmos);
    constexpr int RcMode = RC_flags_storage_2d();
    if constexpr (RcMode & RC_PREAVERAGE) {
        throw std::runtime_error("Dynamic Non-LTE calculation of Gamma incompatible with PREAVERAGE. Try DIR_BY_DIR instead.");
    }

    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    CascadeRaysSubset ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const auto spatial_bounds = mr_block_map.block_map.loop_bounds();

    // NOTE(cmo): Find the atomic index associated with each transition -- we
    // need it grab the right gamma for the GPU. Yes, this is a stall.
    yakl::Array<i32, 1, yakl::memDevice> atom_idx_d("atom_idx", fs_setup.wave_idx.extent(0));
    dex_parallel_for(
        FlatLoop<1>(fs_setup.wave_idx.extent(0)),
        KOKKOS_LAMBDA (int trans) {
            if (fs_setup.trans_type(trans) == FewFreqSetup::TransType::Line) {
                atom_idx_d(trans) = adata.lines(fs_setup.trans_idx(trans)).atom;
            } else {
                atom_idx_d(trans) = adata.continua(fs_setup.trans_idx(trans)).atom;
            }
        }
    );
    Kokkos::fence();
    auto atom_idx = atom_idx_d.createHostCopy();

    for (int iia = la_start; iia < la_end; ++iia) {
        const int ia = atom_idx(iia);
        const auto& Gamma = state.Gamma[ia];
        const auto& psi_star = casc_state.psi_star;
        const auto& I = casc_state.i_cascades[0];
        const auto& incl_quad = state.incl_quad;

        dex_parallel_for(
            "Add Gamma Term",
            FlatLoop<2>(
                spatial_bounds.dim(0),
                spatial_bounds.dim(1)
            ),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord2 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
                ivec2 probe_coord;
                probe_coord(0) = cell_coord.x;
                probe_coord(1) = cell_coord.z;

                for (
                    int phi_idx = ray_subset.start_flat_dirs;
                    phi_idx < ray_subset.start_flat_dirs + ray_subset.num_flat_dirs;
                    phi_idx += 1
                ) {
                    for (
                        int theta_idx = ray_subset.start_incl;
                        theta_idx < ray_subset.start_incl + ray_subset.num_incl;
                        theta_idx += 1
                    ) {
                        const ProbeIndex probe_idx{
                            .coord=probe_coord,
                            .dir=phi_idx,
                            .incl=theta_idx,
                            .wave=iia-subset.la_start
                        };
                        const fp_t intensity = probe_fetch<RcMode>(I, ray_set, probe_idx);
                        const fp_t psi_star_entry = probe_fetch<RcMode>(psi_star, ray_set, probe_idx);

                        const i32 w = iia;

                        AtmosPointParams local_atmos{};
                        local_atmos.temperature = flatmos.temperature(ks);
                        local_atmos.ne = flatmos.ne(ks);
                        local_atmos.vturb = flatmos.vturb(ks);
                        local_atmos.nhtot = flatmos.nh_tot(ks);
                        local_atmos.nh0 = flatmos.nh0(ks);
                        local_atmos.vel = FP(0.0);

                        const int la = fs_setup.wave_idx(w);
                        const fp_t lambda = fs_setup.wavelength(w);
                        const fp_t bandwidth = fs_setup.bandwidth(w);
                        const int kr = fs_setup.trans_idx(w);
                        const auto trans_type = fs_setup.trans_type(w);

                        UV uv;
                        fp_t chi, eta;
                        int ti, tj;
                        if (trans_type == FewFreqSetup::TransType::Line) {
                            using namespace ConstantsFP;
                            // [kJ]
                            const fp_t hnu_4pi = FP(1.0) / lambda * hc_kJ_nm / four_pi;
                            const auto& l = adata.lines(kr);
                            ti = l.i;
                            tj = l.j;
                            // [m2]
                            uv.Vij = hnu_4pi * l.Bij / bandwidth;
                            // [m2]
                            uv.Vji = hnu_4pi * l.Bji / bandwidth;
                            // uv.Vji = FP(0.0);
                            // [kW / (nm sr)]
                            uv.Uji = hnu_4pi * l.Aji / bandwidth;

                            const int offset = adata.level_start(l.atom);
                            const fp_t nj = pops(offset + l.j, ks);
                            const fp_t ni = pops(offset + l.i, ks);
                            eta = nj * uv.Uji;
                            // TODO(cmo): Remove trailing term for stimulated emission?
                            chi = ni * uv.Vij - nj * uv.Vji;
                        } else {
                            using namespace ConstantsFP;
                            const auto& cont = adata.continua(kr);
                            ti = cont.i;
                            tj = cont.j;
                            auto sigma_grid = get_sigma(adata, cont);

                            const int offset = adata.level_start(cont.atom);
                            const fp_t thermal_ratio = n_star(offset + cont.i, ks) / n_star(offset + cont.j, ks) * std::exp(-hc_k_B_nm / (lambda * local_atmos.temperature));

                            // [m2]
                            uv.Vij = sigma_grid.sigma(la - cont.blue_idx);
                            // [m2]
                            uv.Vji = thermal_ratio * uv.Vij;
                            // [kW nm2 / (nm3 m2)] = [kW nm-1] (sr implicit)
                            uv.Uji = twohc2_kW_nm2 / (cube(lambda) * square(lambda) * FP(1e-18)) * uv.Vji;
                            const fp_t nj = pops(offset + cont.j, ks);
                            const fp_t ni = pops(offset + cont.i, ks);
                            eta = nj * uv.Uji;
                            // TODO(cmo): Remove trailing term for stimulated emission?
                            chi = ni * uv.Vij - nj * uv.Vji;
                        }

                        constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
                        const fp_t wlamu = incl_quad.wmuy(theta_idx) / fp_t(c0_dirs_to_average<RcMode>()) * lambda /  hc_4pi * bandwidth;

                        add_to_gamma<true>(GammaAccumState{
                            .eta=eta,
                            .chi=chi,
                            .uv=uv,
                            .I=intensity,
                            .psi_star=psi_star_entry,
                            .wlamu=wlamu,
                            .Gamma=Gamma,
                            .i=ti,
                            .j=tj,
                            .k=ks
                        });
                    }
                }
            }
        );
    }

    Kokkos::fence();

}

void few_freq_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate) {
    JasUnpack(casc_state, mip_chain);

    FewFreqSetup fs_setup = compute_wavelength_setup(state);

    auto pops_dims = state.pops.get_dimensions();
    Fp2d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1));
    compute_lte_pops(&state, lte_scratch);

    // TODO(cmo): Will need to integrate over the BC
    const auto& orig_bc = state.pw_bc;
    State inner_state(state);

    inner_state.pw_bc = PwBc<>{
        .mu_min = orig_bc.mu_min,
        .mu_max = orig_bc.mu_max,
        .mu_step = orig_bc.mu_step,
        .I = Fp2d("pw_bc_I", fs_setup.wave_idx.extent(0), orig_bc.I.extent(1))
    };
    const auto& pw_bc = inner_state.pw_bc;
    dex_parallel_for(
        FlatLoop<2>(fs_setup.wave_idx.extent(0), orig_bc.I.extent(1)),
        KOKKOS_LAMBDA (i32 w, i32 mu) {
            // TODO(cmo): resample this correctly
            pw_bc.I(w, mu) = orig_bc.I(fs_setup.wave_idx(w), mu);
        }
    );
    Kokkos::fence();


    const i32 wave_batch = state.c0_size.wave_batch;
    const i32 num_batches = (fs_setup.wave_idx.extent(0) + wave_batch - 1) / wave_batch;
    // fmt::println("Waves: {} Batches: {}", fs_setup.wave_idx.extent(0), num_batches);
    // fmt::println("Bandwidths");
    auto bw = fs_setup.bandwidth.createHostCopy();
    // for (int i = 0; i < bw.extent(0); ++i) {
    //     fmt::println("{} nm", bw(i));
    // }
    // fmt::println("------------------");
    for (int batch = 0; batch < num_batches; ++batch) {
        const i32 w_start = batch * wave_batch;
        const i32 w_end = std::min((batch + 1) * wave_batch, i32(fs_setup.wave_idx.extent(0)));
        // fmt::println("Batch {}, [{}, {}]", batch, w_start, w_end);

        mip_chain.fill_mip0_atomic_few_freq(state, lte_scratch, fs_setup, w_start, w_end);
        // fmt::println("Fill MIP 0");
        mip_chain.compute_mips(state, w_start, w_end);
        // fmt::println("Fill chain");
        // Do FS
        constexpr int RcModeBc = RC_flags_pack(RcFlags{
            .dynamic = false,
            .preaverage = PREAVERAGE,
            .sample_bc = true,
            .compute_alo = false,
            .dir_by_dir = DIR_BY_DIR
        });
        constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
            .dynamic = false,
            .preaverage = PREAVERAGE,
            .sample_bc = false,
            .compute_alo = false,
            .dir_by_dir = DIR_BY_DIR
        });
        constexpr int RcModeAlo = RC_flags_pack(RcFlags{
            .dynamic = false,
            .preaverage = PREAVERAGE,
            .sample_bc = false,
            .compute_alo = true,
            .dir_by_dir = DIR_BY_DIR
        });
        constexpr int RcStorage = RC_flags_storage_2d();

        // NOTE(cmo): Compute RC FS
        constexpr int num_subsets = subset_tasks_per_cascade<RcStorage>();
        for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
            if (casc_state.psi_star.initialized()) {
                casc_state.psi_star = FP(0.0);
            }
            CascadeCalcSubset subset{
                // NOTE(cmo): This should be entirely for the BC handling
                .la_start=w_start,
                .la_end=w_end,
                .subset_idx=subset_idx
            };

            cascade_i_25d<RcModeBc>(
                inner_state,
                casc_state,
                casc_state.num_cascades,
                subset,
                mip_chain
            );
            // fmt::println("Trace {}", casc_state.num_cascades);
            yakl::fence();
            for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
                cascade_i_25d<RcModeNoBc>(
                    inner_state,
                    casc_state,
                    casc_idx,
                    subset,
                    mip_chain
                );
                // fmt::println("Trace {}", casc_idx);
                yakl::fence();
            }
            if (casc_state.psi_star.initialized() && !lambda_iterate) {
                cascade_i_25d<RcModeAlo>(
                    inner_state,
                    casc_state,
                    0,
                    subset,
                    mip_chain
                );
            } else {
                cascade_i_25d<RcModeNoBc>(
                    inner_state,
                    casc_state,
                    0,
                    subset,
                    mip_chain
                );
            }
            // fmt::println("Trace {}", 0);
            if (casc_state.psi_star.initialized()) {
                few_freq_compute_gamma(
                    inner_state,
                    casc_state,
                    fs_setup,
                    lte_scratch,
                    subset
                );
                // fmt::println("Compute Gamma");
            }
            merge_c0_to_J(
                casc_state,
                state.mr_block_map,
                state.J,
                state.incl_quad,
                w_start,
                w_end
            );
            // fmt::println("Merge C0");
            yakl::fence();
        }
    }
}