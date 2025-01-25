#include "ProfileNormalisation.hpp"

void compute_profile_normalisation(const State& state, const CascadeState& casc_state, bool print_worst_wphi) {
    const auto flatmos = flatten(state.atmos);
    JasUnpack(state, adata, wphi, mr_block_map, incl_quad);
    const auto& profile = state.phi;
    const auto& wavelength = adata.wavelength;
    const int num_cascades = casc_state.num_cascades;
    auto bounds = mr_block_map.block_map.loop_bounds();

    constexpr int RcMode = RC_flags_storage();
    CascadeStorage dims = state.c0_size;
    CascadeRays ray_set = cascade_compute_size<RcMode>(dims, 0);
    // NOTE(cmo): This is pretty thrown-together
    dex_parallel_for(
        "Compute wphi",
        FlatLoop<3>(wphi.extent_int(0), bounds.dim(0), bounds.dim(1)),
        KOKKOS_LAMBDA (int kr, i64 tile_idx, i32 block_idx) {
            using namespace ConstantsFP;
            fp_t entry = FP(0.0);

            IdxGen idx_gen(mr_block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
            ivec2 probe_coord;
            probe_coord(0) = cell_coord.x;
            probe_coord(1) = cell_coord.z;

            const auto& line = adata.lines(kr);

            AtmosPointParams local_atmos;
            local_atmos.temperature = flatmos.temperature(ks);
            local_atmos.ne = flatmos.ne(ks);
            local_atmos.vturb = flatmos.vturb(ks);
            local_atmos.nhtot = flatmos.nh_tot(ks);
            local_atmos.nh0 = flatmos.nh0(ks);

            const fp_t dop_width = doppler_width(line.lambda0, adata.mass(line.atom), local_atmos.temperature, local_atmos.vturb);
            const fp_t gamma = gamma_from_broadening(line, adata.broadening, local_atmos.temperature, local_atmos.ne, local_atmos.nh0);

            for (int la = 0; la < wavelength.extent(0); ++la) {
                if (!line.is_active(la)) {
                    continue;
                }
                const fp_t lambda = wavelength(la);
                fp_t wl_weight = FP(1.0);
                if (la == 0) {
                    wl_weight *= FP(0.5) * (wavelength(1) - wavelength(0));
                } else if (la == wavelength.extent(0) - 1) {
                    wl_weight *= FP(0.5) * (
                        wavelength(wavelength.extent(0) - 1) - wavelength(wavelength.extent(0) - 2)
                    );
                } else {
                    wl_weight *= FP(0.5) * (wavelength(la + 1) - wavelength(la - 1));
                }

                for (int phi_idx = 0; phi_idx < ray_set.num_flat_dirs; ++phi_idx) {
                    for (int theta_idx = 0; theta_idx < ray_set.num_incl; ++theta_idx) {
                        const ProbeIndex probe_idx{
                            .coord=probe_coord,
                            .dir=phi_idx,
                            .incl=theta_idx,
                            .wave=0
                        };
                        RayProps ray = ray_props(ray_set, num_cascades, 0, probe_idx);
                        vec3 mu = inverted_mu(ray, incl_quad.muy(theta_idx));

                        local_atmos.vel = (
                                flatmos.vx(ks) * mu(0)
                                + flatmos.vy(ks) * mu(1)
                                + flatmos.vz(ks) * mu(2)
                        );

                        const fp_t ray_weight = FP(1.0) / fp_t(ray_set.num_flat_dirs) * incl_quad.wmuy(theta_idx);

                        const fp_t a = damping_from_gamma(gamma, lambda, dop_width);
                        const fp_t v = ((lambda - line.lambda0) + (local_atmos.vel * line.lambda0) / c) / dop_width;
                        // [nm-1]
                        const fp_t p = profile(a, v) / (sqrt_pi * dop_width);
                        entry += p * wl_weight * ray_weight;
                    }
                }
            }
            // NOTE(cmo): Store the multiplicative normalisation factor
            if (entry != FP(0.0)) {
                wphi(kr, ks) = FP(1.0) / entry;
            } else {
                wphi(kr, ks) = FP(1.0);
            }
        }
    );
    yakl::fence();

    if (print_worst_wphi) {
        const auto loop = FlatLoop<2>(wphi.extent(0), wphi.extent(1));
        typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<i32, i32>> Reducer;
        typedef Reducer::value_type ReducerType;
        typedef Kokkos::MaxLoc<fp_t, Kokkos::pair<i32, i32>, Kokkos::DefaultExecutionSpace> ReducerDev;

        const auto work_div = balance_parallel_work_division(BalanceLoopArgs{.loop=loop});
        ReducerType max_err_loc;
        Kokkos::parallel_reduce(
            TeamPolicy(work_div.team_count, Kokkos::AUTO()),
            KOKKOS_LAMBDA (const KTeam& team, ReducerType& team_val) {
                const i64 i_base = team.league_rank() * work_div.inner_work_count;
                const i64 i_max = std::min(i_base + work_div.inner_work_count, loop.num_iter);
                const i32 inner_iter_count = i_max - i_base;
                if (inner_iter_count <= 0) {
                    return;
                }
                ReducerType thread_val;
                ReducerDev thread_reducer(thread_val);

                Kokkos::parallel_reduce(
                    InnerRange(team, inner_iter_count),
                    [&] (const int inner_i, ReducerType& inner_val) {
                        auto idxs = loop.unpack(i_base + inner_i);
                        const int kr = idxs[0];
                        const int k = idxs[1];
                        fp_t err = std::abs(FP(1.0) - wphi(kr, k));
                        if (err > inner_val.val) {
                            inner_val.val = err;
                            inner_val.loc = Kokkos::make_pair(kr, k);
                        }
                    },
                    thread_reducer
                );

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    thread_reducer.join(team_val, thread_val);
                });
            },
            Reducer(max_err_loc)
        );

        i32 max_err_k = max_err_loc.loc.second;
        auto spatial_view = Kokkos::subview(wphi, Kokkos::ALL, max_err_k);
        KView<fp_t*> spatial_view_contig("wphi_chunk", spatial_view.extent(0));
        Kokkos::deep_copy(spatial_view_contig, spatial_view);
        auto spatial_view_host = Kokkos::create_mirror_view_and_copy(HostSpace{}, spatial_view_contig);
        std::string output = fmt::format("  Highest error normalisation factors (wphi) @ ks = {}: ", max_err_k);
        for (int kr = 0; kr < spatial_view_host.extent(0); ++kr) {
            output += fmt::format("{:e}", spatial_view_host(kr));
            if (kr != spatial_view_host.extent(0) - 1) {
                output += ", ";
            }
        }
        state.println("{}", output);
    }
}