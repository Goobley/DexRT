#include "MiscSparse.hpp"
#include "RcUtilsModes.hpp"

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

    parallel_for(
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

yakl::Array<u8, 2, yakl::memDevice> reify_active_c0(const BlockMap<BLOCK_SIZE>& block_map) {
    yakl::Array<u8, 2, yakl::memDevice> result(
        "active c0",
        block_map.num_z_tiles * BLOCK_SIZE,
        block_map.num_x_tiles * BLOCK_SIZE
    );
    result = 0;
    yakl::fence();

    parallel_for(
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

void ProbesToCompute::init(
    const CascadeStorage& c0,
    bool sparse_,
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> probes_to_compute
) {
    c0_size = c0;
    sparse = sparse_;
    if (sparse && probes_to_compute.size() == 0) {
        throw std::runtime_error("Sparse config selected, but set of probes_to_compute not provided");
    }
    active_probes = probes_to_compute;
}

void ProbesToCompute::init(
    const State& state,
    int max_cascades
) {
    const bool sparse_calc = state.config.sparse_calculation;
    CascadeStorage c0 = state.c0_size;
    std::vector<yakl::Array<i32, 2, yakl::memDevice>> active_probes;
    if (sparse_calc) {
        active_probes = compute_active_probe_lists(state, max_cascades);
    }
    this->init(c0, sparse_calc, active_probes);
}

i64 ProbesToCompute::num_active_probes(int cascade_idx) const {
    if (sparse) {
        // Throw an exception on out of bounds for easier debugging -- this isn't a fast path
        return active_probes.at(cascade_idx).extent(0);
    }
    CascadeStorage casc = cascade_size(c0_size, cascade_idx);
    return i64(casc.num_probes(0)) * i64(casc.num_probes(1));
}

DeviceProbesToCompute ProbesToCompute::bind(int cascade_idx) const {
    if (sparse) {
        return DeviceProbesToCompute{
            .sparse = true,
            .active_probes = active_probes.at(cascade_idx)
        };
    }

    CascadeStorage casc = cascade_size(c0_size, cascade_idx);
    return DeviceProbesToCompute{
        .sparse = false,
        .num_probes = casc.num_probes
    };
}

// Rehydrates the page from page_idx into qty_page
void rehydrate_page(
    const BlockMap<BLOCK_SIZE>& block_map,
    const Fp2d& quantity,
    const Fp2d& qty_page,
    int page_idx
) {
    qty_page = FP(0.0);
    yakl::fence();

    parallel_for(
        "Rehydrate page",
        block_map.loop_bounds(),
        YAKL_LAMBDA (i64 tile_idx, i32 block_idx) {
            IdxGen idx_gen(block_map);
            i64 ks = idx_gen.loop_idx(tile_idx, block_idx);
            Coord2 coord = idx_gen.loop_coord(tile_idx, block_idx);

            qty_page(coord.z, coord.x) = quantity(page_idx, ks);
        }
    );
    yakl::fence();
}

Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp2d& quantity) {
    const int num_x = block_map.num_x_tiles * BLOCK_SIZE;
    const int num_z = block_map.num_z_tiles * BLOCK_SIZE;
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

Fp3dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp2dHost& quantity) {
    // NOTE(cmo): This is not efficient, it just reuses the GPU machinery,
    // copying a page of CPU memory over at a time
    const int num_x = block_map.num_x_tiles * BLOCK_SIZE;
    const int num_z = block_map.num_z_tiles * BLOCK_SIZE;
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

Fp2dHost rehydrate_sparse_quantity(const BlockMap<BLOCK_SIZE>& block_map, const Fp1d& quantity) {
    Fp2d qtyx1("1 x qty", quantity.data(), 1, quantity.extent(0));
    Fp2d qty_page("qty_page", block_map.num_z_tiles * BLOCK_SIZE, block_map.num_x_tiles * BLOCK_SIZE);
    rehydrate_page(block_map, qtyx1, qty_page, 0);
    Fp2dHost result = qty_page.createHostCopy();
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
    parallel_for(
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
    parallel_for(
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
    fmt::println(
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
            yakl::atomicMax(
                *reinterpret_cast<unsigned long long int*>(&ref),
                val
            );
        };
        parallel_for(
            SimpleBounds<2>(prev_active.extent(0), prev_active.extent(1)),
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
                    probes_to_compute_morton(idx++) = encode_morton_2(Coord2{.x = x, .z = z});
                }
            }
        }
        // NOTE(cmo): These are now being launched in morton order... should be close to tile order
        std::sort(probes_to_compute_morton.begin(), probes_to_compute_morton.end());
        yakl::Array<i32, 2, yakl::memHost> probes_to_compute_h("probes to compute", num_active, 2);
        for (int idx = 0; idx < num_active; ++idx) {
            Coord2 coord = decode_morton_2(probes_to_compute_morton(idx));
            probes_to_compute_h(idx, 0) = coord.x;
            probes_to_compute_h(idx, 1) = coord.z;
        }
        auto probes_to_compute_ci = probes_to_compute_h.createDeviceCopy();
        probes_to_compute.emplace_back(probes_to_compute_ci);
        prev_active = curr_active;
        fmt::println(
            "C{} Active Probes {}/{} ({}%)",
            cascade_idx,
            num_active,
            prev_active.extent(0)*prev_active.extent(1),
            fp_t(num_active) / fp_t(prev_active.extent(0)*prev_active.extent(1)) * FP(100.0)
        );
    }
    return probes_to_compute;
}
