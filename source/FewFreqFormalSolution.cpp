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
        fp_t prev_wl = adata_host.wavelength(cont.blue_idx);
        wavelength.emplace_back(prev_wl);
        wave_idx.emplace_back(cont.blue_idx);
        trans_type.emplace_back(FewFreqSetup::TransType::Continuum);
        trans_idx.emplace_back(kr);
        for (int la = cont.blue_idx; la < cont.red_idx; ++la) {
            if (adata_host.wavelength(la) - prev_wl >= state.config.few_freq.cont_sample_nm || la == cont.red_idx - 1) {
                // NOTE(cmo): This is lagged -- for the previous point
                const fp_t delta = adata_host.wavelength(la) - prev_wl;
                bandwidth.emplace_back(delta);

                wavelength.emplace_back(adata_host.wavelength(la));
                wave_idx.emplace_back(la);
                trans_type.emplace_back(FewFreqSetup::TransType::Continuum);
                trans_idx.emplace_back(kr);

                if (la != cont.red_idx - 1) {
                    prev_wl = adata_host.wavelength(la);
                }
            }
        }
        // NOTE(cmo): Fill in last
        const fp_t delta = adata_host.wavelength(cont.red_idx - 1) - prev_wl;
        bandwidth.emplace_back(delta);
    }

    return FewFreqSetup {
        .wave_idx = yakl::Array<i32, 1, yakl::memHost>("wave_idx", wave_idx.data(), wave_idx.size()).createDeviceCopy(),
        .wavelength = Fp1d("wavelength", wavelength.data(), wavelength.size()).createDeviceCopy(),
        .bandwidth = Fp1d("bandwidth", bandwidth.data(), bandwidth.size()).createDeviceCopy(),
        .trans_type = yakl::Array<FewFreqSetup::TransType, 1, yakl::memHost>("trans_type", trans_type.data(), trans_type.size()).createDeviceCopy(),
        .trans_idx = yakl::Array<i32, 1, yakl::memHost>("trans_idx", trans_idx.data(), trans_idx.size()).createDeviceCopy()
    };
}

void few_freq_formal_sol_rc(const State& state, const CascadeState& casc_state, bool lambda_iterate) {
    JasUnpack(state, mr_block_map, adata_host);
    JasUnpack(casc_state, mip_chain);
    const auto& block_map = mr_block_map.block_map;

    FewFreqSetup fs_setup = compute_wavelength_setup(state);

    auto pops_dims = state.pops.get_dimensions();
    Fp2d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1));
    compute_lte_pops(&state, lte_scratch);

    const i32 wave_batch = state.c0_size.wave_batch;
    const i32 num_batches = (fs_setup.wave_idx.extent(0) + wave_batch - 1) / wave_batch;
    for (int batch = 0; batch < num_batches; ++batch) {
        const i32 w_start = batch * wave_batch;
        const i32 w_end = std::min((batch + 1) * wave_batch, i32(fs_setup.wave_idx.extent(0)));

        mip_chain.fill_mip0_atomic_few_freq(state, lte_scratch, fs_setup, w_start, w_end);
        // TODO(cmo): Will need to integrate over the BC
        const auto& orig_bc = state.pw_bc;
        State inner_state(state);

        inner_state.pw_bc = PwBc<>{
            .mu_min = orig_bc.mu_min,
            .mu_max = orig_bc.mu_max,
            .mu_step = orig_bc.mu_step,
            .I = Fp2d("pw_bc_I", wave_batch, orig_bc.I.extent(1))
        };
        const auto& pw_bc = inner_state.pw_bc;
        dex_parallel_for(
            FlatLoop<2>(wave_batch, orig_bc.I.extent(1)),
            KOKKOS_LAMBDA (i32 w, i32 mu) {
                // TODO(cmo): resample this correctly
                pw_bc.I(w, mu) = orig_bc.I(fs_setup.wave_idx(w_start + w), mu);

            }
        );
        Kokkos::fence();
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
                .la_start=0,
                .la_end=wave_batch,
                .subset_idx=subset_idx
            };

            cascade_i_25d<RcModeBc>(
                state,
                casc_state,
                casc_state.num_cascades,
                subset,
                mip_chain
            );
            yakl::fence();
            for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
                cascade_i_25d<RcModeNoBc>(
                    state,
                    casc_state,
                    casc_idx,
                    subset,
                    mip_chain
                );
                yakl::fence();
            }
            if (casc_state.psi_star.initialized() && !lambda_iterate) {
                cascade_i_25d<RcModeAlo>(
                    state,
                    casc_state,
                    0,
                    subset,
                    mip_chain
                );
            } else {
                cascade_i_25d<RcModeNoBc>(
                    state,
                    casc_state,
                    0,
                    subset,
                    mip_chain
                );
            }
            if (casc_state.psi_star.initialized()) {
                // TODO(cmo): Write the function that goes here
                // dynamic_compute_gamma(
                //     state,
                //     casc_state,
                //     lte_scratch,
                //     subset
                // );
            }
            merge_c0_to_J(
                casc_state,
                state.mr_block_map,
                state.J,
                state.incl_quad,
                w_start,
                w_end
            );
            yakl::fence();
        }
    }
}