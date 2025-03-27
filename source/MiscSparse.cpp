#include "MiscSparse.hpp"
#include "RcUtilsModes.hpp"
#include "State.hpp"
#include "RcUtilsModes3d.hpp"
#include "State3d.hpp"

SparseAtmosphere sparsify_atmosphere(const Atmosphere& atmos, const BlockMap<BLOCK_SIZE>& block_map) {
    i64 num_active_cells = block_map.get_num_active_cells();
    SparseAtmosphere result;
    result.voxel_scale = atmos.voxel_scale;
    result.offset_x = atmos.offset_x;
    result.offset_y = atmos.offset_y;
    result.offset_z = atmos.offset_z;
    result.moving = atmos.moving;
    result.num_x = atmos.temperature.extent(1);
    result.num_y = 1;
    result.num_z = atmos.temperature.extent(0);
    result.temperature = Fp1d("sparse temperature", num_active_cells);
    result.pressure = Fp1d("sparse pressure", num_active_cells);
    result.ne = Fp1d("sparse ne", num_active_cells);
    result.nh_tot = Fp1d("sparse nh_tot", num_active_cells);
    result.nh0 = Fp1d("sparse nh0", num_active_cells);
    result.vturb = Fp1d("sparse vturb", num_active_cells);
    result.vx = Fp1d("sparse vx", num_active_cells);
    result.vy = Fp1d("sparse vy", num_active_cells);
    result.vz = Fp1d("sparse vz", num_active_cells);

    dex_parallel_for(
        "Sparsify atmosphere",
        block_map.loop_bounds(),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            result.temperature(ks) = atmos.temperature(coord.z, coord.x);
            result.pressure(ks) = atmos.pressure(coord.z, coord.x);
            result.ne(ks) = atmos.ne(coord.z, coord.x);
            result.nh_tot(ks) = atmos.nh_tot(coord.z, coord.x);
            result.nh0(ks) = atmos.nh0(coord.z, coord.x);
            result.vturb(ks) = atmos.vturb(coord.z, coord.x);
            result.vx(ks) = atmos.vx(coord.z, coord.x);
            result.vy(ks) = atmos.vy(coord.z, coord.x);
            result.vz(ks) = atmos.vz(coord.z, coord.x);
        }
    );
    yakl::fence();
    return result;
}

SparseAtmosphere sparsify_atmosphere(
    const AtmosphereNd<3, yakl::memHost>& atmos,
    const BlockMap<BLOCK_SIZE_3D, 3>& block_map
) {
    i64 num_active_cells = block_map.get_num_active_cells();
    SparseAtmosphere result;
    result.voxel_scale = atmos.voxel_scale;
    result.offset_x = atmos.offset_x;
    result.offset_y = atmos.offset_y;
    result.offset_z = atmos.offset_z;
    result.moving = atmos.moving;
    result.num_x = atmos.temperature.extent(2);
    result.num_y = atmos.temperature.extent(1);
    result.num_z = atmos.temperature.extent(0);
    result.temperature = Fp1d("sparse temperature", num_active_cells);
    result.pressure = Fp1d("sparse pressure", num_active_cells);
    result.ne = Fp1d("sparse ne", num_active_cells);
    result.nh_tot = Fp1d("sparse nh_tot", num_active_cells);
    result.nh0 = Fp1d("sparse nh0", num_active_cells);
    result.vturb = Fp1d("sparse vturb", num_active_cells);
    result.vx = Fp1d("sparse vx", num_active_cells);
    result.vy = Fp1d("sparse vy", num_active_cells);
    result.vz = Fp1d("sparse vz", num_active_cells);

    auto copy_and_sparsify_field = [&](const Fp1d& dest, const Fp3dHost& source) {
        Fp3d src = source.createDeviceCopy();
        dex_parallel_for(
            "Sparsify field",
            block_map.loop_bounds(),
            KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
                IdxGen3d idx_gen(block_map);
                i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
                Coord3 coord = idx_gen.loop_coord(tile_idx, block_idx);
                dest(ks) = src(coord.z, coord.y, coord.x);
            }
        );
    };

    copy_and_sparsify_field(result.temperature, atmos.temperature);
    copy_and_sparsify_field(result.pressure, atmos.pressure);
    copy_and_sparsify_field(result.ne, atmos.ne);
    copy_and_sparsify_field(result.nh_tot, atmos.nh_tot);
    copy_and_sparsify_field(result.nh0, atmos.nh0);
    copy_and_sparsify_field(result.vturb, atmos.vturb);
    copy_and_sparsify_field(result.vx, atmos.vx);
    copy_and_sparsify_field(result.vy, atmos.vy);
    copy_and_sparsify_field(result.vz, atmos.vz);

    Kokkos::fence();
    return result;
}

yakl::Array<u8, 2, yakl::memDevice> reify_active_c0(const BlockMap<BLOCK_SIZE>& block_map) {
    yakl::Array<u8, 2, yakl::memDevice> result(
        "active c0",
        block_map.num_z_tiles() * BLOCK_SIZE,
        block_map.num_x_tiles() * BLOCK_SIZE
    );
    result = 0;
    yakl::fence();

    dex_parallel_for(
        "Eval active",
        block_map.loop_bounds(),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(block_map);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            result(coord.z, coord.x) = 1;
        }
    );
    yakl::fence();
    return result;
}

// Rehydrates the page from page_idx into qty_page
template <int BLOCK_SIZE, int NumDim>
void rehydrate_page(
    const BlockMap<BLOCK_SIZE, NumDim>& block_map,
    const Fp2d& quantity,
    const yakl::Array<fp_t, NumDim, yakl::memDevice>& qty_page,
    int page_idx
) {
    qty_page = FP(0.0);
    yakl::fence();

    using IdxGen_t = std::conditional_t<NumDim == 2, IdxGen, IdxGen3d>;

    dex_parallel_for(
        "Rehydrate page",
        block_map.loop_bounds(),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen_t idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord<NumDim> coord = idx_gen.loop_coord(tile_idx, block_idx);

            JasUse(qty_page, quantity, page_idx);
            if constexpr (NumDim == 2) {
                qty_page(coord.z, coord.x) = quantity(page_idx, ks);
            } else {
                qty_page(coord.z, coord.y, coord.x) = quantity(page_idx, ks);
            }
        }
    );
    yakl::fence();
}

Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp2d& quantity) {
    const int num_x = block_map.num_x_tiles() * block_map.BLOCK_SIZE;
    const int num_z = block_map.num_z_tiles() * block_map.BLOCK_SIZE;
    Fp3dHost result(
        quantity.myname,
        quantity.extent(0),
        num_z,
        num_x
    );
    Fp2d qty_page("qty_page", num_z, num_x);

    for (int n = 0; n < quantity.extent(0); ++n) {
        rehydrate_page(block_map, quantity, qty_page, n);
        Fp2dHost qty_page_host = qty_page.createHostCopy();

        for (int z = 0; z < num_z; ++z) {
            for (int x = 0; x < num_x; ++x) {
                result(n, z, x) = qty_page_host(z, x);
            }
        }
    }
    return result;
}

Fp4dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE_3D, 3>& block_map, const Fp2d& quantity) {
    const int num_x = block_map.num_x_tiles() * block_map.BLOCK_SIZE;
    const int num_y = block_map.num_y_tiles() * block_map.BLOCK_SIZE;
    const int num_z = block_map.num_z_tiles() * block_map.BLOCK_SIZE;
    Fp4dHost result(
        quantity.myname,
        quantity.extent(0),
        num_z,
        num_y,
        num_x
    );
    Fp3d qty_page("qty_page", num_z, num_y, num_x);

    for (int n = 0; n < quantity.extent(0); ++n) {
        rehydrate_page(block_map, quantity, qty_page, n);
        Fp3dHost qty_page_host = qty_page.createHostCopy();

        for (int z = 0; z < num_z; ++z) {
            for (int y = 0; z < num_y; ++y) {
                for (int x = 0; x < num_x; ++x) {
                    result(n, z, y, x) = qty_page_host(z, y, x);
                }
            }
        }
    }
    return result;
}

Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp2dHost& quantity) {
    // NOTE(cmo): This is not efficient, it just reuses the GPU machinery,
    // copying a page of CPU memory over at a time
    const int num_x = block_map.num_x_tiles() * block_map.BLOCK_SIZE;
    const int num_z = block_map.num_z_tiles() * block_map.BLOCK_SIZE;
    Fp3dHost result(
        quantity.myname,
        quantity.extent(0),
        num_z,
        num_x
    );
    Fp2d qty_page("qty_page", num_z, num_x);
    Fp2d qty_gpu("qty_gpu", 1, quantity.extent(1));

    for (int n = 0; n < quantity.extent(0); ++n) {
        qty_gpu = Fp2dHost("qty_slice", &quantity(n, 0), 1, quantity.extent(1)).createDeviceCopy();
        rehydrate_page(block_map, qty_gpu, qty_page, 0);
        Fp2dHost qty_page_host = qty_page.createHostCopy();

        for (int z = 0; z < num_z; ++z) {
            for (int x = 0; x < num_x; ++x) {
                result(n, z, x) = qty_page_host(z, x);
            }
        }
    }
    return result;
}

Fp4dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE_3D, 3>& block_map, const Fp2dHost& quantity) {
    // NOTE(cmo): This is not efficient, it just reuses the GPU machinery,
    // copying a page of CPU memory over at a time
    const int num_x = block_map.num_x_tiles() * block_map.BLOCK_SIZE;
    const int num_y = block_map.num_y_tiles() * block_map.BLOCK_SIZE;
    const int num_z = block_map.num_z_tiles() * block_map.BLOCK_SIZE;
    Fp4dHost result(
        quantity.myname,
        quantity.extent(0),
        num_z,
        num_y,
        num_x
    );
    Fp3d qty_page("qty_page", num_z, num_y, num_x);
    Fp2d qty_gpu("qty_gpu", 1, quantity.extent(1));

    for (int n = 0; n < quantity.extent(0); ++n) {
        qty_gpu = Fp2dHost("qty_slice", &quantity(n, 0), 1, quantity.extent(1)).createDeviceCopy();
        rehydrate_page(block_map, qty_gpu, qty_page, 0);
        Fp3dHost qty_page_host = qty_page.createHostCopy();

        for (int z = 0; z < num_z; ++z) {
            for (int y = 0; y < num_y; ++y) {
                for (int x = 0; x < num_x; ++x) {
                    result(n, z, y, x) = qty_page_host(z, y, x);
                }
            }
        }
    }
    return result;
}

Fp2dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp1d& quantity) {
    Fp2d qtyx1("1 x qty", quantity.data(), 1, quantity.extent(0));
    Fp2d qty_page(
        "qty_page",
        block_map.num_z_tiles() * block_map.BLOCK_SIZE,
        block_map.num_x_tiles() * block_map.BLOCK_SIZE
    );
    rehydrate_page(block_map, qtyx1, qty_page, 0);
    Fp2dHost result = qty_page.createHostCopy();
    return result;
}

Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE_3D, 3>& block_map, const Fp1d& quantity) {
    Fp2d qtyx1("1 x qty", quantity.data(), 1, quantity.extent(0));
    Fp3d qty_page(
        "qty_page",
        block_map.num_z_tiles() * block_map.BLOCK_SIZE,
        block_map.num_y_tiles() * block_map.BLOCK_SIZE,
        block_map.num_x_tiles() * block_map.BLOCK_SIZE
    );
    rehydrate_page(block_map, qtyx1, qty_page, 0);
    Fp3dHost result = qty_page.createHostCopy();
    return result;
}

std::vector<yakl::Array<i32, 2, yakl::memDevice>> compute_active_probe_lists(const State& state, int max_cascades) {
    // TODO(cmo): This is a poor strategy for 3D, but simple for now. To be done properly in parallel we need to do some stream compaction. e.g. thrust::copy_if
    // Really this function is backwards. We can loop over each probe of i+1 and
    // check the dependents in i, in parallel. The merge process remains the
    // same.
    JasUnpack(state, mr_block_map);
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> probes_to_compute;
    probes_to_compute.reserve(max_cascades + 1);

    yakl::Array<u64, 2, yakl::memDevice> prev_active("active c0", state.atmos.num_z, state.atmos.num_x);
    prev_active = 0;
    yakl::fence();
    dex_parallel_for(
        mr_block_map.block_map.loop_bounds(),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            prev_active(coord.z, coord.x) = 1;
        }
    );
    yakl::fence();
    u64 num_active = mr_block_map.get_num_active_cells();

    auto prev_active_h = prev_active.createHostCopy();
    yakl::fence();
    yakl::Array<i32, 2, yakl::memDevice> probes_to_compute_c0("c0 to compute", num_active, 2);
    dex_parallel_for(
        mr_block_map.block_map.loop_bounds(),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(mr_block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);
            probes_to_compute_c0(ks, 0) = coord.x;
            probes_to_compute_c0(ks, 1) = coord.z;
        }
    );
    probes_to_compute.emplace_back(probes_to_compute_c0);
    state.println(
        "C0 Active Probes {}/{} ({}%)",
        num_active,
        prev_active.extent(0)*prev_active.extent(1),
        fp_t(num_active) / fp_t(prev_active.extent(0)*prev_active.extent(1)) * FP(100.0)
    );


    for (int cascade_idx = 1; cascade_idx <= max_cascades; ++cascade_idx) {
        CascadeStorage dims = cascade_size(state.c0_size, cascade_idx);
        yakl::Array<u64, 2, yakl::memDevice> curr_active(
            "casc_active",
            dims.num_probes(1),
            dims.num_probes(0)
        );
        curr_active = 0;
        yakl::fence();
        auto my_atomic_max = YAKL_LAMBDA (u64& ref, unsigned long long int val) {
            Kokkos::atomic_max(
                reinterpret_cast<unsigned long long int*>(&ref),
                val
            );
        };
        dex_parallel_for(
            FlatLoop<2>(prev_active.extent(0), prev_active.extent(1)),
            YAKL_LAMBDA (int z, int x) {
                int z_bc = std::max(int((z - 1) / 2), 0);
                int x_bc = std::max(int((x - 1) / 2), 0);
                const bool z_clamp = (z_bc == 0) || (z_bc == (curr_active.extent(0) - 1));
                const bool x_clamp = (x_bc == 0) || (x_bc == (curr_active.extent(1) - 1));

                if (!prev_active(z, x)) {
                    return;
                }

                // NOTE(cmo): Atomically set the (up-to) 4 valid
                // probes for this active probe of cascade_idx-1
                my_atomic_max(curr_active(z_bc, x_bc), 1);
                if (!x_clamp) {
                    my_atomic_max(curr_active(z_bc, x_bc+1), 1);
                }
                if (!z_clamp) {
                    my_atomic_max(curr_active(z_bc+1, x_bc), 1);
                }
                if (!(z_clamp || x_clamp)) {
                    my_atomic_max(curr_active(z_bc+1, x_bc+1), 1);
                }
            }
        );
        yakl::fence();
        i64 num_active = yakl::intrinsics::sum(curr_active);
        auto curr_active_h = curr_active.createHostCopy();
        yakl::fence();
        yakl::Array<u32, 1, yakl::memHost> probes_to_compute_morton("probes to compute morton", num_active);
        i32 idx = 0;
        for (int z = 0; z < curr_active_h.extent(0); ++z) {
            for (int x = 0; x < curr_active_h.extent(1); ++x) {
                if (curr_active_h(z, x)) {
                    probes_to_compute_morton(idx++) = encode_morton(Coord2{.x = x, .z = z});
                }
            }
        }
        // NOTE(cmo): These are now being launched in morton order... should be close to tile order
        std::sort(probes_to_compute_morton.begin(), probes_to_compute_morton.end());
        yakl::Array<i32, 2, yakl::memHost> probes_to_compute_h("probes to compute", num_active, 2);
        for (int idx = 0; idx < num_active; ++idx) {
            Coord2 coord = decode_morton<2>(probes_to_compute_morton(idx));
            probes_to_compute_h(idx, 0) = coord.x;
            probes_to_compute_h(idx, 1) = coord.z;
        }
        auto probes_to_compute_ci = probes_to_compute_h.createDeviceCopy();
        probes_to_compute.emplace_back(probes_to_compute_ci);
        prev_active = curr_active;
        state.println(
            "C{} Active Probes {}/{} ({}%)",
            cascade_idx,
            num_active,
            prev_active.extent(0)*prev_active.extent(1),
            fp_t(num_active) / fp_t(prev_active.extent(0)*prev_active.extent(1)) * FP(100.0)
        );
    }
    return probes_to_compute;
}

std::vector<yakl::Array<Coord3, 1, yakl::memDevice>> compute_active_probe_lists(const State3d& state, int max_cascades) {
    // NOTE(cmo): This is the strategy discussed in the 2D model. It makes a lot
    // more sense, and 2D should be migrated.
    JasUnpack(state, mr_block_map, c0_size);
    std::vector<yakl::Array<Coord3, 1, yakl::memDevice>> probes_to_compute;
    probes_to_compute.reserve(max_cascades + 1);

    i64 num_active = mr_block_map.get_num_active_cells();
    i64 total_num_probes = c0_size.num_probes(0) * c0_size.num_probes(1) * c0_size.num_probes(2);
    yakl::Array<Coord3, 1, yakl::memDevice> probes_to_compute_c0("C0 to compute", num_active);
    dex_parallel_for(
        "Compute C0 active",
        mr_block_map.block_map.loop_bounds(),
        KOKKOS_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen3d idx_gen(mr_block_map);
            Coord3 coord = idx_gen.loop_coord(tile_idx, block_idx);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            probes_to_compute_c0(ks) = coord;
        }
    );
    Kokkos::fence();
    probes_to_compute.emplace_back(probes_to_compute_c0);
    state.println(
        "C0 Active Probes {}/{} ({}%)",
        num_active,
        total_num_probes,
        fp_t(num_active) / fp_t(total_num_probes) * FP(100.0)
    );

    // Hoist this, but will only be created for C1 onwards (not C0)
    yakl::Array<i8, 3, yakl::memDevice> prev_active;

    // NOTE(cmo): Repeat this process for the other cascades
    for (int cascade_idx = 1; cascade_idx <= max_cascades; ++cascade_idx) {
        CascadeStorage3d prev_dims = cascade_size(state.c0_size, cascade_idx-1);
        CascadeStorage3d dims = cascade_size(state.c0_size, cascade_idx);
        yakl::Array<i8, 3, yakl::memDevice> next_probes(
            "cn active",
            dims.num_probes(2),
            dims.num_probes(1),
            dims.num_probes(0)
        );
        yakl::Array<Coord3, 1, yakl::memDevice> next_probe_coords("Cn probes", next_probes.size());
        next_probes = i8(0);
        Kokkos::fence();
        dex_parallel_for(
            "Compute Cn active",
            FlatLoop<3>(dims.num_probes(2), dims.num_probes(1), dims.num_probes(0)),
            KOKKOS_LAMBDA (int zn, int yn, int xn) {
                int zp = 2 * zn;
                int yp = 2 * yn;
                int xp = 2 * xn;
                IdxGen3d idx_gen(mr_block_map);

                auto check_lower_cascade = [&]() -> bool {
                    // NOTE(cmo): On each axis, check in range [x0 - 1, x0 + 2]. If any
                    // are active, this one is also active.
                    for (int z = zp - 1; z < zp + 3; ++z) {
                        for (int y = yp - 1; y < yp + 3; ++y) {
                            for (int x = xp - 1; x < xp + 3; ++x) {
                                // NOTE(cmo): clamp access
                                Coord3 coord {
                                    .x = std::min(std::max(x, 0), prev_dims.num_probes(0) - 1),
                                    .y = std::min(std::max(y, 0), prev_dims.num_probes(1) - 1),
                                    .z = std::min(std::max(z, 0), prev_dims.num_probes(2) - 1)
                                };

                                // NOTE(cmo): Special handling for C1 to leverage the blockmap over C0
                                if (cascade_idx == 1) {
                                    if (idx_gen.has_leaves(coord)) {
                                        return true;
                                    }
                                } else {
                                    if (prev_active(coord.z, coord.y, coord.x)) {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                    return false;
                };

                if (check_lower_cascade()) {
                    next_probes(zn, yn, xn) = 1;
                }

                Coord3 cn_coord {
                    .x = xn,
                    .y = yn,
                    .z = zn
                };
                // NOTE(cmo): This is the array we will stream compact
                next_probe_coords((zn * dims.num_probes(1) + yn) * dims.num_probes(0) + xn) = cn_coord;
            }
        );
        Kokkos::fence();

        i64 num_active_cn;
        dex_parallel_reduce(
            "Compute number Cn active",
            FlatLoop<3>(dims.num_probes(2), dims.num_probes(1), dims.num_probes(0)),
            KOKKOS_LAMBDA (int zn, int yn, int xn, i64& num_active) {
                if (next_probes(zn, yn, xn)) {
                    num_active += 1;
                }
            },
            Kokkos::Sum<i64>(num_active_cn)
        );

        yakl::Array<Coord3, 1, yakl::memDevice> probes_to_compute_cn("Cn to compute", num_active_cn);
        if constexpr (false) {
            auto probes_to_compute_host = probes_to_compute_cn.createHostObject();
            auto next_probe_coords_host = next_probe_coords.createHostCopy();
            auto next_probes_host = next_probes.createHostCopy();
            std::copy_if(
                next_probe_coords_host.data(),
                next_probe_coords_host.data() + next_probe_coords_host.size(),
                probes_to_compute_host.data(),
                [&] (const Coord3& c) {
                    return next_probes_host(c.z, c.y, c.x);
                }
            );
            probes_to_compute_cn = probes_to_compute_host.createDeviceCopy();
        } else {
            Kokkos::Experimental::copy_if(
                "Compact C1 coords",
                DefaultExecutionSpace{},
                KView<Coord3*>(next_probe_coords.data(), next_probe_coords.size()),
                KView<Coord3*>(probes_to_compute_cn.data(), probes_to_compute_cn.size()),
                KOKKOS_LAMBDA (const Coord3& c) {
                    return next_probes(c.z, c.y, c.x);
                }
            );
            Kokkos::fence();
        }
        // NOTE(cmo): Kokkos sort is producing errors from thrust when included
        // Kokkos::sort(
        //     KView<Coord3*>(probes_to_compute_cn.data(), probes_to_compute_cn.size()),
        //     KOKKOS_LAMBDA (const Coord3& l, const Coord3& r) {
        //         // NOTE(cmo): This may overflow on very large models, but it's only an optimisation.
        //         return encode_morton<3>(l) < encode_morton<3>(r);
        //     }
        // );
        constexpr bool copy_to_host_and_sort = true;
        if constexpr (copy_to_host_and_sort) {
            auto probes_to_compute_host = probes_to_compute_cn.createHostCopy();
            std::sort(
                probes_to_compute_host.data(),
                probes_to_compute_host.data() + probes_to_compute_host.size(),
                [] (const Coord3& l, const Coord3& r) {
                    // NOTE(cmo): This may overflow on very large models, but it's only an optimisation.
                    return encode_morton<3>(l) < encode_morton<3>(r);
                }
            );
            probes_to_compute_cn = probes_to_compute_host.createDeviceCopy();
        }
        Kokkos::fence();
        probes_to_compute.emplace_back(probes_to_compute_cn);
        prev_active = next_probes;
        state.println(
            "C{} Active Probes {}/{} ({}%)",
            cascade_idx,
            num_active_cn,
            prev_active.size(),
            fp_t(num_active_cn) / fp_t(prev_active.size()) * FP(100.0)
        );
    }
    return probes_to_compute;
}
