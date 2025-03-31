#include "DynamicFormalSolution3d.hpp"
#include "CascadeState.hpp"
#include "Mipmaps3d.hpp"
#include "RayMarching.hpp" // only for merge_intervals
#include "BoundaryDispatch.hpp"
#include "GammaMatrix.hpp"

template <typename BcType=void>
KOKKOS_INLINE_FUNCTION
BcType get_bc(const DeviceBoundaries& bounds) {

}

template <>
KOKKOS_INLINE_FUNCTION
ZeroBc get_bc(const DeviceBoundaries& bounds) {
    return bounds.zero_bc;
}

template <>
KOKKOS_INLINE_FUNCTION
PwBc<> get_bc(const DeviceBoundaries& bounds) {
    return bounds.pw_bc;
}

template <typename DynamicState=void>
KOKKOS_INLINE_FUNCTION
DynamicState get_dyn_state_3d(
    int la,
    const SparseAtmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const MultiResMipChain3d& mip_chain
) {
    return DynamicState{};
}

struct Raymarch3dDynamicState {
    const yakl::Array<const u16, 1, yakl::memDevice> active_set;
    const SparseAtmosphere& atmos;
    const AtomicData<fp_t>& adata;
    const VoigtProfile<fp_t, false>& profile;
    const Fp1d& nh0;
    const Fp2d& n;
};

template <>
KOKKOS_INLINE_FUNCTION
Raymarch3dDynamicState get_dyn_state_3d(
    int la,
    const SparseAtmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const MultiResMipChain3d& mip_chain
) {
    return Raymarch3dDynamicState{
        .active_set = slice_active_set(adata, la),
        .atmos = atmos,
        .adata = adata,
        .profile = profile,
        .nh0 = atmos.nh0,
        .n = flat_pops
    };
}

struct Raymarch3dDynamicCavState {
    const yakl::SArray<i32, 1, CORE_AND_VOIGT_MAX_LINES_3D> active_set;
    const VoigtProfile<fp_t, false>& profile;
    const AtomicData<fp_t>& adata;
};

template <>
KOKKOS_INLINE_FUNCTION
Raymarch3dDynamicCavState get_dyn_state_3d(
    int la,
    const SparseAtmosphere& atmos,
    const AtomicData<fp_t>& adata,
    const VoigtProfile<fp_t>& profile,
    const Fp2d& flat_pops,
    const MultiResMipChain3d& mip_chain
) {
    auto basic_a_set = slice_active_set(adata, la);
    yakl::SArray<i32, 1, CORE_AND_VOIGT_MAX_LINES_3D> local_active_set; // In krl indices
    const auto& krl_mapping = mip_chain.cav_data.active_set_mapping;
    int l_idx = 0;
    for (int a = 0; a < basic_a_set.extent(0); ++a) {
        i32 kr = basic_a_set(a);
        for (int krl = 0; krl < CORE_AND_VOIGT_MAX_LINES_3D; ++krl) {
            if (krl_mapping(krl) == kr) {
                local_active_set(l_idx++) = krl;
            }
        }
    }
    if (l_idx < CORE_AND_VOIGT_MAX_LINES_3D) {
        local_active_set(l_idx) = -1;
    }
    return Raymarch3dDynamicCavState{
        .active_set = local_active_set,
        .profile = profile,
        .adata = adata
    };
}


template <int RcMode=0>
struct RcDynamicState3d {
    typedef typename std::conditional_t<
        RcMode & RC_DYNAMIC && (LINE_SCHEME_3D == LineCoeffCalc::CoreAndVoigt),
        Raymarch3dDynamicCavState,
        std::conditional_t<
            RcMode & RC_DYNAMIC,
            Raymarch3dDynamicState,
            DexEmpty
        >
    > type;
};


void dynamic_compute_gamma(
    const State3d& state,
    const CascadeState3d& casc_state,
    const Fp2d& lte_scratch,
    int la,
    int subset_idx
) {
    JasUnpack(state, phi, pops, adata, wphi, mr_block_map);
    const auto flatmos = flatten<const fp_t>(state.atmos);
    constexpr int RcMode = RC_flags_storage_3d();
    const auto& wavelength = adata.wavelength;

    CascadeStorage3d dims = state.c0_size;
    CascadeRays3d ray_set = cascade_compute_size<RcMode>(dims, 0);
    CascadeRaysSubset3d ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);
    const int num_cascades = casc_state.num_cascades;
    const auto spatial_bounds = mr_block_map.block_map.loop_bounds();
    for (int ia = 0; ia < state.adata_host.num_level.extent(0); ++ia) {
        const auto& Gamma = state.Gamma[ia];
        const auto& alo = casc_state.alo;
        const auto& I = casc_state.i_cascades[0];
        dex_parallel_for(
            "Compute Gamma",
            FlatLoop<4>(
                spatial_bounds.dim(0),
                spatial_bounds.dim(1),
                ray_subset.num_polar_rays,
                ray_subset.num_az_rays
            ),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx, int theta_idx, int phi_idx) {
                IdxGen3d idx_gen(mr_block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord3 cell_coord = idx_gen.loop_coord(tile_idx, block_idx);
                ivec3 probe_coord;
                probe_coord(0) = cell_coord.x;
                probe_coord(1) = cell_coord.y;
                probe_coord(2) = cell_coord.z;

                theta_idx += ray_subset.start_polar_rays;
                phi_idx += ray_subset.start_az_rays;
                const ProbeIndex3d probe_idx{
                    .coord=probe_coord,
                    .polar = theta_idx,
                    .az = phi_idx
                };
                RaySegment<3> ray = probe_ray(ray_set, num_cascades, 0, probe_idx);
                const fp_t intensity = probe_fetch<RcMode>(I, ray_set, probe_idx);
                const fp_t alo_entry = probe_fetch<RcMode>(alo, ray_set, probe_idx);

                const fp_t lambda = wavelength(la);
                using namespace ConstantsFP;
                constexpr bool include_hc_4pi = false;
                constexpr fp_t hc_4pi = hc_kJ_nm / four_pi;
                fp_t hnu_4pi = FP(1.0);
                if (include_hc_4pi) {
                    hnu_4pi = hc_kJ_nm / (four_pi);
                }
                fp_t wl_weight = lambda / hnu_4pi;
                if (la == 0) {
                    wl_weight *= FP(0.5) * (wavelength(1) - wavelength(0));
                } else if (la == wavelength.extent(0) - 1) {
                    wl_weight *= FP(0.5) * (
                        wavelength(wavelength.extent(0) - 1) - wavelength(wavelength.extent(0) - 2)
                    );
                } else {
                    wl_weight *= FP(0.5) * (wavelength(la + 1) - wavelength(la - 1));
                }
                const fp_t wl_ray_weight = wl_weight / fp_t(ray_set.num_az_rays * ray_set.num_polar_rays);
                AtmosPointParams local_atmos;
                local_atmos.temperature = flatmos.temperature(ks);
                local_atmos.ne = flatmos.ne(ks);
                local_atmos.vturb = flatmos.vturb(ks);
                local_atmos.nhtot = flatmos.nh_tot(ks);
                local_atmos.nh0 = flatmos.nh0(ks);
                local_atmos.vel = (
                        flatmos.vx(ks) * ray.d(0)
                        + flatmos.vy(ks) * ray.d(1)
                        + flatmos.vz(ks) * ray.d(2)
                );
                const int kr_base = adata.line_start(ia);
                for (int kr_atom = 0; kr_atom < adata.num_line(ia); ++kr_atom) {
                    const int kr = kr_base + kr_atom;
                    const auto& l = adata.lines(kr);
                    if (!l.is_active(la)) {
                        continue;
                    }
                    const UV uv = compute_uv_line(
                        EmisOpacState<>{
                            .adata = adata,
                            .profile = phi,
                            .la = la,
                            .n = pops,
                            .n_star_scratch = lte_scratch,
                            .k = ks,
                            .atmos = local_atmos
                        },
                        kr,
                        UvOptions {
                            .include_hc_4pi = include_hc_4pi
                        }
                    );

                    const int offset = adata.level_start(ia);
                    fp_t eta = pops(offset + l.j, ks) * uv.Uji;
                    fp_t chi = pops(offset + l.i, ks) * uv.Vij - pops(offset + l.j, ks) * uv.Vji;
                    if (!include_hc_4pi) {
                        eta *= hc_4pi;
                        chi *= hc_4pi;
                    }
                    chi += FP(1e-20);

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .alo = alo_entry,
                        .wlamu = wl_ray_weight * wphi(kr, ks),
                        .Gamma = Gamma,
                        .i = l.i,
                        .j = l.j,
                        .k = ks
                    });
                }
                const int kr_base_c = adata.cont_start(ia);
                for (int kr_atom = 0; kr_atom < adata.num_cont(ia); ++kr_atom) {
                    const int kr = kr_base_c + kr_atom;
                    const auto& cont = adata.continua(kr);
                    if (!cont.is_active(la)) {
                        continue;
                    }

                    const UV uv = compute_uv_cont(
                        EmisOpacState<>{
                            .adata = adata,
                            .profile = phi,
                            .la = la,
                            .n = pops,
                            .n_star_scratch = lte_scratch,
                            .k = ks,
                            .atmos = local_atmos
                        },
                        kr,
                        UvOptions {
                            .include_hc_4pi = include_hc_4pi
                        }
                    );

                    const int offset = adata.level_start(ia);
                    fp_t eta = pops(offset + cont.j, ks) * uv.Uji;
                    fp_t chi = pops(offset + cont.i, ks) * uv.Vij - pops(offset + cont.j, ks) * uv.Vji;
                    if (!include_hc_4pi) {
                        eta *= hc_4pi;
                        chi *= hc_4pi;
                    }
                    chi += FP(1e-20);

                    add_to_gamma<true>(GammaAccumState{
                        .eta = eta,
                        .chi = chi,
                        .uv = uv,
                        .I = intensity,
                        .alo = alo_entry,
                        .wlamu = wl_ray_weight,
                        .Gamma = Gamma,
                        .i = cont.i,
                        .j = cont.j,
                        .k = ks
                    });
                }
            }
        );
    }
    Kokkos::fence();
}

template <typename Bc, typename DynamicState>
struct Raymarch3dArgs {
    const ProbeIndex3d& this_probe;
    const DeviceCascadeState3d& casc_state;
    const MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>& mr_block_map;
    const RaySegment<3>& ray;
    const fp_t distance_scale;
    const MultiResMipChain3d& mip_chain;
    const i32 max_mip_to_sample;
    Bc bc;
    DynamicState dyn_state;
    int la;
    vec3 offset;
};

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> multi_level_dda_raymarch_3d(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, mr_block_map, ray, distance_scale, mip_chain, la, bc, offset, dyn_state);
    constexpr bool dynamic = (RcMode & RC_DYNAMIC);
    // constexpr bool dynamic_interp = dynamic && std::is_same_v<DynamicState, Raymarch3dDynamicInterpState>;
    constexpr bool dynamic_cav = dynamic && std::is_same_v<DynamicState, Raymarch3dDynamicCavState>;
    RadianceInterval<Alo> result;

    MRIdxGen3d idx_gen(mr_block_map);
    auto s = MultiLevelDDA<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3>(idx_gen);
    bool start_clipped;
    const bool marcher = s.init(ray, args.max_mip_to_sample, &start_clipped);
    constexpr bool always_sample_bc = (RcMode & RC_SAMPLE_BC) && LAST_CASCADE_TO_INFTY && !(RcMode & RC_LINE_SWEEP);
    const bool ray_starts_outside = (RcMode & RC_SAMPLE_BC) && (!marcher || start_clipped);
    if (always_sample_bc || ray_starts_outside) {
        // NOTE(cmo): Check the ray is going up along z.
        if ((ray.d(2) > FP(0.0)) && la != -1) {
            vec3 pos;
            pos(0) = ray.o(0) * distance_scale + offset(0);
            pos(1) = ray.o(1) * distance_scale + offset(1);
            pos(2) = ray.o(2) * distance_scale + offset(2);

            fp_t I_sample = sample_boundary(bc, la, pos, ray.d);
            result.I = I_sample;
        }
    }
    if (!marcher) {
        return result;
    }
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> eta_k(
        mip_chain.emis.data(),
        mip_chain.emis.extent(0)
    );
    KView<fp_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>> chi_k(
        mip_chain.opac.data(),
        mip_chain.opac.extent(0)
    );

    fp_t lambda;
    if constexpr (dynamic && std::is_same_v<DynamicState, Raymarch3dDynamicCavState>) {
        lambda = dyn_state.adata.wavelength(la);
    }

    fp_t eta_s = FP(0.0), chi_s = FP(1e-20), one_m_edt = FP(0.0);
    do {
        one_m_edt = FP(0.0);
        if (s.can_sample()) {
            i64 ks = idx_gen.idx(
                s.current_mip_level,
                Coord3{.x = s.curr_coord(0), .y = s.curr_coord(1), .z = s.curr_coord(2)}
            );
            if constexpr (dynamic_cav) {
                JasUnpack(dyn_state, active_set, profile, adata);
                eta_s = mip_chain.emis(ks);
                chi_s = mip_chain.opac(ks) + FP(1e-15);

                const fp_t vel = (
                    mip_chain.vx(ks) * ray.d(0)
                    + mip_chain.vy(ks) * ray.d(1)
                    + mip_chain.vz(ks) * ray.d(2)
                );
                CavEmisOpacState emis_opac_state {
                    .ks = ks,
                    .krl = 0,
                    .wave = 0,
                    .lambda = lambda,
                    .vel = vel,
                    .phi = profile
                };

                #pragma unroll
                for (int kri = 0; kri < CORE_AND_VOIGT_MAX_LINES_3D; ++kri) {
                    i32 krl = active_set(kri);
                    if (krl < 0) {
                        break;
                    }
                    emis_opac_state.krl = krl;
                    i32 kr = mip_chain.cav_data.active_set_mapping(krl);
                    EmisOpac eta_chi = mip_chain.cav_data.emis_opac(
                        emis_opac_state
                    );
                    eta_s += eta_chi.eta;
                    chi_s += eta_chi.chi;
                }
            } else {
                eta_s = eta_k(ks);
                chi_s = chi_k(ks) + FP(1e-15);
                if constexpr (dynamic) {
                    const SparseAtmosphere& atmos = dyn_state.atmos;
                    if (
                        mip_chain.classic_data.dynamic_opac(ks)
                        && dyn_state.active_set.extent(0) > 0
                    ) {
                        const fp_t vel = (
                            atmos.vx(ks) * ray.d(0)
                            + atmos.vy(ks) * ray.d(1)
                            + atmos.vz(ks) * ray.d(2)
                        );
                        AtmosPointParams local_atmos{
                            .temperature = atmos.temperature.get_data()[ks],
                            .ne = atmos.ne.get_data()[ks],
                            .vturb = atmos.vturb.get_data()[ks],
                            .nhtot = atmos.nh_tot.get_data()[ks],
                            .vel = vel,
                            .nh0 = dyn_state.nh0.get_data()[ks]
                        };
                        auto lines = emis_opac(
                            EmisOpacState<fp_t>{
                                .adata = dyn_state.adata,
                                .profile = dyn_state.profile,
                                .la = la,
                                .n = dyn_state.n,
                                .k = ks,
                                .atmos = local_atmos,
                                .active_set = dyn_state.active_set,
                                .mode = EmisOpacMode::DynamicOnly
                            }
                        );
                        eta_s += lines.eta;
                        chi_s += lines.chi;
                    }
                }
            }

            fp_t tau = chi_s * s.dt * distance_scale;
            fp_t source_fn = eta_s / chi_s;
            fp_t edt = std::exp(-tau);
            one_m_edt = -std::expm1(-tau);
            result.tau += tau;
            result.I = result.I * edt + source_fn * one_m_edt;
        }

    } while (s.step_through_grid());

    if constexpr ((RcMode & RC_COMPUTE_ALO) && !std::is_same_v<Alo, DexEmpty>) {
        result.alo = std::max(one_m_edt, FP(0.0));
    }
    return result;
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
KOKKOS_INLINE_FUNCTION RadianceInterval<Alo> march_and_merge_average_interval_3d(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, this_probe, casc_state);
    RadianceInterval<Alo> ri = multi_level_dda_raymarch_3d<RcMode>(args);

    RadianceInterval<Alo> interp;
    if (casc_state.upper_I.initialized()) {
        JasUnpack(casc_state, upper_I, upper_tau, casc_dims, upper_dims);

        TexelsPerRay3d upper_tex = upper_texels_per_ray_3d<RcMode>(casc_state.n);
        int upper_polar_start_idx = this_probe.polar * upper_dims.num_polar_rays / casc_dims.num_polar_rays;
        int upper_az_start_idx = this_probe.az * upper_dims.num_az_rays / casc_dims.num_az_rays;
        const fp_t ray_weight = FP(1.0) / fp_t(upper_tex.az * upper_tex.polar);

        TrilinearCorner base = trilinear_corner(this_probe.coord);
        vec<8> weights = trilinear_weights(base);
        for (int tri_idx = 0; tri_idx < 8; ++tri_idx) {
            ivec3 upper_coord = trilinear_coord(base, upper_dims.num_probes, tri_idx);
            for (
                int upper_polar_idx = upper_polar_start_idx;
                upper_polar_idx < upper_polar_start_idx + upper_tex.polar;
                ++upper_polar_idx
            ) {
                for (
                    int upper_az_idx = upper_az_start_idx;
                    upper_az_idx < upper_az_start_idx + upper_tex.az;
                    ++upper_az_idx
                ) {
                    ProbeIndex3d upper_probe {
                        .coord = upper_coord,
                        .polar = upper_polar_idx,
                        .az = upper_az_idx
                    };
                    i64 lin_idx = probe_linear_index<RcMode>(upper_dims, upper_probe);
                    interp.I += ray_weight * weights(tri_idx) * upper_I(lin_idx);
                    if constexpr (STORE_TAU_CASCADES) {
                        interp.tau += ray_weight * weights(tri_idx) * upper_tau(lin_idx);
                    }
                }
            }
        }
    }
    return merge_intervals(ri, interp);
}

template <
    int RcMode=0,
    typename Bc,
    typename DynamicState,
    typename Alo=std::conditional_t<bool(RcMode & RC_COMPUTE_ALO), fp_t, DexEmpty>
>
YAKL_INLINE RadianceInterval<Alo> march_and_merge_trilinear_interval_3d(
    const Raymarch3dArgs<Bc, DynamicState>& args
) {
    JasUnpack(args, this_probe, casc_state);

    RadianceInterval<Alo> interp;
    if (casc_state.upper_I.initialized()) {
        JasUnpack(casc_state, upper_I, upper_tau, casc_dims, upper_dims);
        CascadeRays3d casc_rays = cascade_storage_to_rays<RcMode>(casc_dims);
        CascadeRays3d upper_casc_rays = cascade_storage_to_rays<RcMode>(upper_dims);

        TexelsPerRay3d upper_tex = upper_texels_per_ray_3d<RcMode>(casc_state.n);
        int upper_polar_start_idx = this_probe.polar * upper_dims.num_polar_rays / casc_dims.num_polar_rays;
        int upper_az_start_idx = this_probe.az * upper_dims.num_az_rays / casc_dims.num_az_rays;
        const fp_t ray_weight = FP(1.0) / fp_t(upper_tex.az * upper_tex.polar);

        TrilinearCorner base = trilinear_corner(this_probe.coord);
        vec<8> weights = trilinear_weights(base);
        for (int tri_idx = 0; tri_idx < 8; ++tri_idx) {
            ivec3 upper_coord = trilinear_coord(base, upper_dims.num_probes, tri_idx);
            RaySegment<3> tri_ray = trilinear_probe_ray(
                casc_rays,
                upper_casc_rays,
                casc_state.num_cascades,
                casc_state.n,
                this_probe,
                upper_coord
            );
            RadianceInterval<Alo> ri = multi_level_dda_raymarch_3d<RcMode>(
                Raymarch3dArgs {
                    .this_probe = args.this_probe,
                    .casc_state = args.casc_state,
                    .mr_block_map = args.mr_block_map,
                    .ray = tri_ray,
                    .distance_scale = args.distance_scale,
                    .mip_chain = args.mip_chain,
                    .max_mip_to_sample = args.max_mip_to_sample
                }
            );

            RadianceInterval<Alo> upper_interp{};
            for (
                int upper_polar_idx = upper_polar_start_idx;
                upper_polar_idx < upper_polar_start_idx + upper_tex.polar;
                ++upper_polar_idx
            ) {
                for (
                    int upper_az_idx = upper_az_start_idx;
                    upper_az_idx < upper_az_start_idx + upper_tex.az;
                    ++upper_az_idx
                ) {
                    ProbeIndex3d upper_probe {
                        .coord = upper_coord,
                        .polar = upper_polar_idx,
                        .az = upper_az_idx
                    };
                    i64 lin_idx = probe_linear_index<RcMode>(upper_dims, upper_probe);
                    upper_interp.I += ray_weight * upper_I(lin_idx);
                    if constexpr (STORE_TAU_CASCADES) {
                        upper_interp.tau += ray_weight * upper_tau(lin_idx);
                    }
                }
            }
            RadianceInterval<Alo> merged = merge_intervals(ri, upper_interp);
            interp.I += weights(tri_idx) * merged.I;
            if constexpr (STORE_TAU_CASCADES) {
                interp.tau += weights(tri_idx) * merged.tau;
            }
        }
    } else {
        interp = multi_level_dda_raymarch_3d(args);
    }
    return interp;
}

static void merge_c0_to_J_3d(const State3d& state, const CascadeState3d& casc_state, int la, fp_t ray_weight=FP(-1.0)) {
    constexpr int RcMode = RC_flags_storage_3d();
    const bool sparse = casc_state.probes_to_compute.sparse;
    const CascadeStorage3d c0_dims = casc_state.probes_to_compute.c0_size;
    CascadeRays3d ray_set = cascade_compute_size<RcMode>(state.c0_size, 0);
    if (ray_weight < FP(0.0)) {
        ray_weight = FP(1.0) / fp_t(ray_set.num_az_rays * ray_set.num_polar_rays);
    }

    const auto& c0 = casc_state.i_cascades[0];
    const auto& J = state.J;

    JasUnpack(state, mr_block_map);
    const FlatLoop<2> spatial_bounds = mr_block_map.block_map.loop_bounds();

    const bool J_slice = (J.extent(0) == 1);
    DeviceProbesToCompute<3> probes_to_compute = casc_state.probes_to_compute.bind(0);

    dex_parallel_for(
        "Final cascade to J",
        FlatLoop<3>(probes_to_compute.num_active_probes(), c0_dims.num_polar_rays, c0_dims.num_az_rays),
        KOKKOS_LAMBDA (i64 k, int theta_idx, int phi_idx) {
            ivec3 probe_coord = probes_to_compute(k);
            ProbeStorageIndex3d this_probe {
                .coord = probe_coord,
                .polar = theta_idx,
                .az = phi_idx
            };

            i64 ks;
            if (sparse) {
                IdxGen3d idx_gen(mr_block_map);
                ks = idx_gen.idx(Coord3{.x = probe_coord(0), .y = probe_coord(1), .z = probe_coord(2)});
            } else {
                ks = (probe_coord(2) * c0_dims.num_probes(1) + probe_coord(1)) * c0_dims.num_probes(0) + probe_coord(0);
            }

            int inner_la = J_slice ? 0 : la;
            const fp_t sample = probe_fetch<RcMode>(c0, c0_dims, this_probe);
            Kokkos::atomic_add(&J(inner_la, ks), ray_weight * sample);
        }
    );
    Kokkos::fence();
}

template <int RcMode>
void compute_cascade_i_3d(const State3d& state, const CascadeState3d& casc_state, int la, int subset_idx, int casc_idx) {
    JasUnpack(state, atmos, phi, adata, pops, mr_block_map);
    const auto& mip_chain = casc_state.mip_chain;
    constexpr bool compute_alo = RcMode & RC_COMPUTE_ALO;
    using Alo = std::conditional_t<compute_alo, fp_t, DexEmpty>;
    typedef typename RcDynamicState3d<RcMode>::type DynamicState;

    const fp_t distance_scale = atmos.voxel_scale;
    CascadeStorage3d dims = cascade_size(state.c0_size, casc_idx);
    CascadeStorage3d upper_dims = cascade_size(state.c0_size, casc_idx+1);
    CascadeRays3d ray_set = cascade_compute_size<RcMode>(state.c0_size, casc_idx);
    CascadeRaysSubset3d ray_subset = nth_rays_subset<RcMode>(ray_set, subset_idx);

    CascadeIdxs lookup = cascade_indices(casc_state, casc_idx);
    Fp1d i_cascade_i = casc_state.i_cascades[lookup.i];
    Fp1d tau_cascade_i = casc_state.tau_cascades[lookup.i];
    FpConst1d i_cascade_ip, tau_cascade_ip;
    if (lookup.ip != -1) {
        i_cascade_ip = casc_state.i_cascades[lookup.ip];
        tau_cascade_ip = casc_state.tau_cascades[lookup.ip];
    }

    const int max_mip_to_sample = std::min(
        state.config.mip_config.mip_levels[casc_idx],
        mip_chain.max_mip_factor
    );

    DeviceCascadeState3d dev_casc_state {
        .num_cascades = casc_state.num_cascades,
        .n = casc_idx,
        .casc_dims = dims,
        .upper_dims = upper_dims,
        .cascade_I = i_cascade_i,
        .cascade_tau = tau_cascade_i,
        .upper_I = i_cascade_ip,
        .upper_tau = tau_cascade_ip
    };
    if constexpr (compute_alo) {
        dev_casc_state.alo = casc_state.alo;
    }
    DeviceBoundaries boundaries_h{
        .boundary = state.boundary,
        .zero_bc = state.zero_bc,
        .pw_bc = state.pw_bc
    };
    auto offset = get_offsets(atmos);

    std::string name = fmt::format("Cascade {}", casc_idx);
    yakl::timer_start(name);

    i64 spatial_bounds = casc_state.probes_to_compute.num_active_probes(casc_idx);
    DeviceProbesToCompute<3> probe_coord_lookup = casc_state.probes_to_compute.bind(casc_idx);

    dex_parallel_for(
        "RC Loop 3D",
        FlatLoop<3>(spatial_bounds, ray_subset.num_polar_rays, ray_subset.num_az_rays),
        KOKKOS_LAMBDA (i64 flat_probe_idx, int theta_idx, int phi_idx) {
            ivec3 probe_coord = probe_coord_lookup(flat_probe_idx);

            theta_idx += ray_subset.start_polar_rays;
            phi_idx += ray_subset.start_az_rays;
            ProbeIndex3d probe_idx {
                .coord=probe_coord,
                .polar = theta_idx,
                .az = phi_idx
            };

            RaySegment<3> ray = probe_ray(ray_set, dev_casc_state.num_cascades, casc_idx, probe_idx);

            // compute_ri
            constexpr bool trilinear_fix = false;
            RadianceInterval<Alo> ri;

            const auto& boundaries = boundaries_h;

            auto dispatch = [&]<typename BcType, typename DynamicState>(const BcType& bc, const DynamicState& ds) {
                Raymarch3dArgs<BcType, DynamicState> args {
                    .this_probe = probe_idx,
                    .casc_state = dev_casc_state,
                    .mr_block_map = mr_block_map,
                    .ray = ray,
                    .distance_scale = distance_scale,
                    .mip_chain = mip_chain,
                    .max_mip_to_sample = max_mip_to_sample,
                    .bc = get_bc<BcType>(boundaries),
                    .dyn_state = ds,
                    .la = la,
                    .offset = offset
                };
                if constexpr (trilinear_fix) {
                    ri = march_and_merge_trilinear_interval_3d<RcMode>(
                        args
                    );
                } else {
                    ri = march_and_merge_average_interval_3d<RcMode>(
                        args
                    );
                }
            };
            DynamicState dyn_state = get_dyn_state_3d<DynamicState>(
                la,
                atmos,
                adata,
                phi,
                pops,
                mip_chain
            );
            if constexpr (RcMode & RC_SAMPLE_BC) {
                switch (boundaries.boundary) {
                    case BoundaryType::Zero: {
                        dispatch(ZeroBc{}, dyn_state);
                    } break;
                    case BoundaryType::Promweaver:{
                        dispatch(PwBc<>{}, dyn_state);
                    } break;
                    default: {
                        Kokkos::abort("Unknown BC type");
                    }
                }
            } else {
                dispatch(ZeroBc{}, dyn_state);
            }
            i64 lin_idx = probe_linear_index<RcMode>(dims, probe_idx);
            dev_casc_state.cascade_I(lin_idx) = ri.I;
            if constexpr (STORE_TAU_CASCADES) {
                dev_casc_state.cascade_tau(lin_idx) = ri.tau;
            }
            constexpr bool dev_compute_alo = bool(RcMode & RC_COMPUTE_ALO);
            if constexpr (dev_compute_alo) {
                dev_casc_state.alo(lin_idx) = ri.alo;
            }
        }
    );
    Kokkos::fence();

    yakl::timer_stop(name);
}

void dynamic_formal_sol_rc_3d_subset(
    const State3d& state,
    const CascadeState3d& casc_state,
    bool lambda_iterate,
    int la,
    int subset_idx
) {
    constexpr int RcModeBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = false,
        .sample_bc = true,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR_3D
    });
    constexpr int RcModeNoBc = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = false,
        .sample_bc = false,
        .compute_alo = false,
        .dir_by_dir = DIR_BY_DIR_3D
    });
    constexpr int RcModeAlo = RC_flags_pack(RcFlags{
        .dynamic = true,
        .preaverage = false,
        .sample_bc = false,
        .compute_alo = true,
        .dir_by_dir = DIR_BY_DIR_3D
    });

    if (casc_state.num_cascades > 0) {
        compute_cascade_i_3d<RcModeBc>(
            state,
            casc_state,
            la,
            subset_idx,
            casc_state.num_cascades
        );
    }
    for (int casc_idx = casc_state.num_cascades - 1; casc_idx >= 1; --casc_idx) {
        compute_cascade_i_3d<RcModeNoBc>(
            state,
            casc_state,
            la,
            subset_idx,
            casc_idx
        );
    }
    if (casc_state.alo.initialized() && !lambda_iterate) {
        compute_cascade_i_3d<RcModeAlo>(
            state,
            casc_state,
            la,
            subset_idx,
            0
        );
    } else {
        compute_cascade_i_3d<RcModeNoBc>(
            state,
            casc_state,
            la,
            subset_idx,
            0
        );
    }
}

void dynamic_formal_sol_rc_3d(const State3d& state, const CascadeState3d& casc_state, bool lambda_iterate, int la) {
    JasUnpack(casc_state, mip_chain);

    // TODO(cmo): This scratch space isn't ideal right now - we will get rid of
    // it, for now, trust the pool allocator
    auto pops_dims = state.pops.get_dimensions();
    Fp2d lte_scratch("lte_scratch", pops_dims(0), pops_dims(1));

    mip_chain.fill_mip0_atomic(state, lte_scratch, la);
    mip_chain.compute_mips(state, la);

    constexpr int RcStorage = RC_flags_storage_3d();

    constexpr int num_subsets = subset_tasks_per_cascade_3d<RcStorage>();
    for (int subset_idx = 0; subset_idx < num_subsets; ++subset_idx) {
        if (casc_state.alo.initialized()) {
            casc_state.alo = FP(0.0);
        }
        dynamic_formal_sol_rc_3d_subset(state, casc_state, lambda_iterate, la, subset_idx);
        if (casc_state.alo.initialized()) {
            dynamic_compute_gamma(
                state,
                casc_state,
                lte_scratch,
                la,
                subset_idx
            );
        }
        merge_c0_to_J_3d(
            state,
            casc_state,
            la
        );
        Kokkos::fence();
    }
}