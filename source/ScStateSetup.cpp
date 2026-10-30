#include "ScStateSetup.hpp"
#include "StateSetup.hpp"
#include "ScIo.hpp"
#include "ScOutput.hpp"
#include "MiscSparse.hpp"
#include "BlockMap.hpp"
#include "Atmosphere.hpp"
#include <fmt/core.h>

namespace {

typedef yakl::Array<f32, 3, yakl::memHost> F32Host3d;

/// Activity of this rank's rows, by the global tile criterion (a tile is
/// active if any cell is at or below threshold). Slabs are whole tile rows,
/// so every tile is local. T is [z, y, x] (2D: y = 1).
ScActiveMaskHost sc_local_activity(const F32Host3d& T, i32 tile, fp_t threshold) {
    const i32 nz = T.extent(0), ny = T.extent(1), nx = T.extent(2);
    const i32 tile_y = ny == 1 ? 1 : tile;
    if (nz % tile || nx % tile || ny % tile_y) {
        throw std::runtime_error(fmt::format("SC atmospheres need complete tiles ({} cells)", tile));
    }
    const i32 nxt = nx / tile, nyt = ny / tile_y;
    ScActiveMaskHost tiles("sc_local_tiles", nz / tile, i64(nyt) * nxt);
    for (i32 zt = 0; zt < nz / tile; ++zt) {
        for (i32 yt = 0; yt < nyt; ++yt) {
            for (i32 xt = 0; xt < nxt; ++xt) {
                bool active = threshold == FP(0.0);
                for (i32 z = 0; z < tile && !active; ++z) {
                    for (i32 y = 0; y < tile_y; ++y) {
                        for (i32 x = 0; x < tile; ++x) {
                            active |= T(zt * tile + z, yt * tile_y + y, xt * tile + x) <= threshold;
                        }
                    }
                }
                tiles(zt, yt * nxt + xt) = active;
            }
        }
    }
    return sc_active_mask_from_tiles(tiles, 0, nz, tile, ny);
}

/// The owned rows of a dense [z, (y,) x] atmosphere (read by rank 0 and
/// streamed to their owners), in the slab index space (void cells zeroed).
SparseAtmosphere sc_load_atmos_window(ScState* state, const DexrtConfig& config) {
    const bool three_d = get_dexrt_dimensionality() == 3;
    const i32 tile = three_d ? SC_TILE_3D : BLOCK_SIZE;
    const ScSlab& slab = state->slab;
    ScInFile in(state->mpi_state, config.atmos_path);
    const i32 nx = in.dim("x"), ny = three_d ? in.dim("y") : 1;
    const i32 z0 = slab.z_start, nz = slab.num_z();
    F32Host3d buf("sc_atmos_window", nz, ny, nx);
    auto read = [&](const char* name) {
        const auto s = in.shape(name);
        if (s.size() != (three_d ? 3 : 2) || s[0] != slab.global_num_z || s.back() != nx) {
            throw std::runtime_error(fmt::format("SC atmosphere field {} must be dense [z, {}x]", name, three_d ? "y, " : ""));
        }
        in.read_block(name, 0, z0, nz, buf.data());
    };

    read("temperature");
    state->cells.init(sc_local_activity(buf, tile, config.threshold_temperature), z0, SC_CELL_GROUP, ny);
    const ScCells& cells = state->cells;
    const auto mask = cells.active.createHostCopy();
    const i64 off = cells.owned_offset();
    // NOTE(cmo): The active cells of buf, in the dense slab layout.
    auto dense = [&]() -> Fp1d {
        Fp1dHost values("sc_atmos_field", cells.num_cells());
        values = FP(0.0);
        for (i64 k = 0; k < cells.num_owned_cells(); ++k) {
            if (mask.data()[off + k]) {
                values(off + k) = buf.data()[k];
            }
        }
        return values.createDeviceCopy();
    };
    auto field = [&](const char* name, bool optional = false) -> Fp1d {
        if (optional && !in.has_var(name)) {
            return {};
        }
        read(name);
        return dense();
    };

    SparseAtmosphere result;
    result.voxel_scale = in.read_scalar<f32>("voxel_scale");
    result.offset_x = in.has_var("offset_x") ? in.read_scalar<f32>("offset_x") : FP(0.0);
    result.offset_y = in.has_var("offset_y") ? in.read_scalar<f32>("offset_y") : FP(0.0);
    result.offset_z = in.has_var("offset_z") ? in.read_scalar<f32>("offset_z") : FP(0.0);
    result.num_x = nx;
    result.num_y = ny;
    result.num_z = nz;
    result.temperature = dense();
    result.pressure = field("pressure");
    result.ne = field("ne");
    result.nh_tot = field("nh_tot");
    result.vturb = field("vturb");
    // NOTE(cmo): Moving if any owned cell (active or not) has |v| > 10 m/s.
    std::vector<f32> v2(i64(nz) * ny * nx, 0.0f);
    for (auto [name, target] : {std::pair{"vx", &result.vx}, std::pair{"vy", &result.vy}, std::pair{"vz", &result.vz}}) {
        *target = field(name);
        for (i64 i = 0; i < v2.size(); ++i) {
            v2[i] += square(buf.data()[i]);
        }
    }
    const f32 max_v2 = v2.empty() ? 0.0f : *std::max_element(v2.begin(), v2.end());
    result.moving = std::sqrt(max_v2) > FP(10.0);
    result.e_int = field("e_int", true);
    result.nh0 = Fp1d("nh0", cells.num_cells());
    result.nh0 = FP(0.0);
    yakl::fence();
    return result;
}

/// The owned rows of a dense [lead, z, (y,) x] field, as [lead, ks] (void
/// cells zeroed).
Fp2d sc_read_cells(ScInFile& in, const ScState& state, const std::string& name, i64 lead) {
    const auto& cells = state.cells;
    const auto l = sc_dense_layout(cells, state.slab, {"lead"}, {lead});
    if (in.shape(name) != l.shape) {
        throw std::runtime_error(fmt::format("SC: {} must be dense [{}, z, (y,) x] matching the atmosphere", name, lead));
    }
    yakl::Array<fp_t, 3, yakl::memHost> rows("sc_read_dense", lead, cells.num_z, cells.plane_size());
    in.read_block(name, l.split, cells.z_start, cells.num_z, rows.data());
    return cells.from_owned_rows(rows).createDeviceCopy();
}

/// As sc_read_cells, for a dense [z, (y,) x] field.
Fp1d sc_read_cells_scalar(ScInFile& in, const ScState& state, const std::string& name) {
    const auto& cells = state.cells;
    const auto l = sc_dense_layout(cells, state.slab, {}, {});
    if (in.shape(name) != l.shape) {
        throw std::runtime_error(fmt::format("SC: {} must be dense [z, (y,) x] matching the atmosphere", name));
    }
    yakl::Array<fp_t, 3, yakl::memHost> rows("sc_read_dense", 1, cells.num_z, cells.plane_size());
    in.read_block(name, 0, cells.z_start, cells.num_z, rows.data());
    return cells.from_owned_rows(rows).createDeviceCopy().reshape(cells.num_cells());
}

} // namespace

void init_atmos_atoms(ScState* st, const DexrtConfig& config) {
    if (!(config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte)) {
        return;
    }

    ScState& state = *st;
    init_atomic_data(st, config);
    int sparse = 0;
    {
        ScInFile in(state.mpi_state, config.atmos_path);
        in.att_int("sparse", &sparse);
    }
    if (!sparse) {
        state.atmos = sc_load_atmos_window(&state, config);
    } else {
        if (get_dexrt_dimensionality() == 3) {
            throw std::runtime_error("dexrt_sc_3d needs a dense atmosphere file.");
        }
        if (state.mpi_state.num_ranks > 1) {
            throw std::runtime_error("Sparse atmosphere files are only supported by dexrt_sc on one rank.");
        }
        // NOTE(cmo): A sparse (block map) atmosphere, in the slab index space.
        MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE> mr_block_map;
        state.atmos = mr_block_map.init_and_sparsify_atmos(config.atmos_path, config.threshold_temperature, 0);
        const auto mask = reify_active_c0(mr_block_map.block_map).createHostCopy();
        state.cells.init(mask);
        const ScCells cells = state.cells;
        const auto list = cells.list();
        const auto legacy = mr_block_map.block_map;
        auto reorder = [&](const Fp1d& a) -> Fp1d {
            if (!a.initialized()) {
                return {};
            }
            Fp1d out(a.label(), cells.num_cells());
            out = FP(0.0);
            dex_parallel_for(
                "SC import sparse atmosphere",
                FlatLoop<1>(list.num()),
                YAKL_LAMBDA(i64 i) {
                    IdxGen index(legacy);
                    const i64 ks = list(i);
                    const Coord3 c = cells.coord(ks);
                    out(ks) = a(index.idx(Coord2{.x = c.x, .z = c.z}));
                });
            return out;
        };
        auto& a = state.atmos;
        a.temperature = reorder(a.temperature);
        a.pressure = reorder(a.pressure);
        a.ne = reorder(a.ne);
        a.nh_tot = reorder(a.nh_tot);
        a.nh0 = reorder(a.nh0);
        a.vturb = reorder(a.vturb);
        a.vx = reorder(a.vx);
        a.vy = reorder(a.vy);
        a.vz = reorder(a.vz);
        a.e_int = reorder(a.e_int);
        yakl::fence();
    }


    init_atom_storage(st, config);
    // NOTE(cmo): The void and halo cells stay benign (never updated).
    state.pops = FP(0.0);
    if (state.wphi.initialized()) {
        state.wphi.fill(FP(1.0));
    }
    yakl::fence();
}

void init_given_emis_opac(ScState* st, const DexrtConfig& config) {
    if (config.mode != DexrtMode::GivenFs) {
        return;
    }
    ScInFile in(st->mpi_state, config.atmos_path);
    const ScSlab& slab = st->slab;
    // NOTE(cmo): 3D (dexrt_sc_3d): [z, y, x, wavelength], held as [z, y * x,
    // wavelength] (the SC plane layout).
    const bool three_d = get_dexrt_dimensionality() == 3;
    const int x_dim = in.dim("x");
    const int y_dim = three_d ? in.dim("y") : 1;
    const int z_dim = slab.num_z();
    const int wave_dim = in.dim("wavelength");
    const int plane_dim = x_dim * y_dim;

    // NOTE(cmo): With the halo rows of the ScCells layout (filled by ScHalo).
    const int rows = z_dim + 2 * SC_MEDIUM_HALO;
    const i64 halo_offset = i64(SC_MEDIUM_HALO) * plane_dim * wave_dim;
    F32Host3d eta("eta", rows, plane_dim, wave_dim);
    F32Host3d chi("chi", rows, plane_dim, wave_dim);
    eta = 0.0f;
    chi = 0.0f;
    st->given_state.num_y = y_dim;
    in.read_block("eta", 0, slab.z_start, z_dim, eta.data() + halo_offset);
    in.read_block("chi", 0, slab.z_start, z_dim, chi.data() + halo_offset);

    f32 voxel_scale = FP(1.0);
    if (in.has_var("voxel_scale")) {
        voxel_scale = in.read_scalar<f32>("voxel_scale");
    }
    st->atmos.voxel_scale = voxel_scale;
    st->given_state.voxel_scale = voxel_scale;
    st->println("Scale: {} m", st->atmos.voxel_scale);

#ifdef DEXRT_SINGLE_PREC
    st->given_state.emis = eta.createDeviceCopy();
    st->given_state.opac = chi.createDeviceCopy();
#else
    auto etad = eta.createDeviceCopy();
    auto chid = chi.createDeviceCopy();
    Fp3d emis("eta", rows, plane_dim, wave_dim);
    Fp3d opac("chi", rows, plane_dim, wave_dim);
    yakl::fence();
    dex_parallel_for(
        "Convert f32->f64",
        FlatLoop<3>(rows, plane_dim, wave_dim),
        YAKL_LAMBDA (int z, int x, int la) {
            emis(z, x, la) = etad(z, x, la);
        }
    );
    dex_parallel_for(
        "Convert f32->f64",
        FlatLoop<3>(rows, plane_dim, wave_dim),
        YAKL_LAMBDA (int z, int x, int la) {
            opac(z, x, la) = chid(z, x, la);
        }
    );
    yakl::fence();
    st->given_state.emis = emis;
    st->given_state.opac = opac;
#endif

    // NOTE(cmo): Only zero boundaries are supported here.
    st->boundary = BoundaryType::Zero;
    yakl::fence();
}

void allocate_J(ScState* state) {
    JasUnpack((*state), config, adata);
    // NOTE(cmo): The owned rows.
    const i64 num_cells = state->cells.num_owned_cells();
    i32 wave_dim = adata.wavelength.extent(0);
    if (config.mode == DexrtMode::GivenFs) {
        wave_dim = state->given_state.emis.extent(2);
    }
    state->J_cpu = Rad16ArrayHost();
    state->J_halo_cpu = Rad16ArrayHost();
    if (config.output.J || state->background.enabled) {
        state->J_cpu = Rad16ArrayHost("J_cpu", wave_dim, num_cells);
        state->J_cpu.fill(FP(0.0));
    }
    if (state->background.enabled) {
        state->J_halo_cpu = Rad16ArrayHost("J_halo_cpu", wave_dim, 2 * state->cells.owned_offset());
        state->J_halo_cpu.fill(FP(0.0));
    }
}

void allocate_rad_loss(ScState* state) {
    const auto& config = state->config;
    if (config.rad_loss == RadLossType::None) {
        return;
    }
    if (config.mode == DexrtMode::GivenFs) {
        throw std::runtime_error("SC rad_loss needs atoms (wavelength bins), not GivenFs");
    }
    const i64 num_cells = state->cells.num_owned_cells();
    const i32 rows = config.rad_loss == RadLossType::Integrated ? 1 : state->adata_host.wavelength.extent(0);
    state->rad_loss_integrated = yakl::Array<RadLossFp, 1, yakl::memDevice>("sc_rad_loss_integrated", num_cells);
    state->rad_loss_cpu = yakl::Array<RadLossFp, 2, yakl::memHost>("sc_rad_loss_cpu", rows, num_cells);
    state->rad_loss_integrated = RadLossFp(0.0);
    state->rad_loss_cpu = RadLossFp(0.0);
    yakl::fence();
}

/// Loads the populations (and ne, nh_tot, temperature, pressure, PRD rho and J
/// for the background, if present) from a dense snapshot. Returns the current
/// iteration number.
int handle_restart(ScState* st, const std::string& restart_path) {
    ScState& state(*st);
    ScInFile in(state.mpi_state, restart_path);
    sc_read_cells(in, state, "pops", state.pops.extent(0)).deep_copy_to(state.pops);
    for (auto [name, target] : {
        std::pair{"ne", &state.atmos.ne},
        std::pair{"nh_tot", &state.atmos.nh_tot},
        std::pair{"temperature", &state.atmos.temperature},
        std::pair{"pressure", &state.atmos.pressure}
    }) {
        if (in.has_var(name)) {
            sc_read_cells_scalar(in, state, name).deep_copy_to(*target);
        }
    }
    if (state.prd.enabled() && in.has_var("prd_rho")) {
        state.prd.rho = Rad16Array::from_fp(sc_read_cells(in, state, "prd_rho", state.prd.num_rows));
    }
    if (state.background.enabled && in.has_var("J")) {
        const auto& cells = state.cells;
        const i32 waves = state.J_cpu.extent(0);
        const auto l = sc_dense_layout(cells, state.slab, {"wavelength"}, {waves});
        if (in.shape("J") != l.shape) {
            throw std::runtime_error("SC: J must be dense [wavelength, z, (y,) x] matching the atmosphere");
        }
        int encoded = 0;
        in.att_int("J_rad16", &encoded);
        if (encoded && !HALF_RADIATION_STORAGE) {
            throw std::runtime_error("SC: Rad16 snapshot J requires DEX_HALF_RADIATION_STORAGE=ON");
        }
        if (encoded || !HALF_RADIATION_STORAGE) {
            // The file contains this build's native storage; receive directly into J.
            in.read_block("J", l.split, cells.z_start, cells.num_z, state.J_cpu.data.data());
        } else {
            // Legacy snapshots contain decoded floating-point J.
            Fp2dHost J("sc_restart_J", waves, cells.num_owned_cells());
            in.read_block("J", l.split, cells.z_start, cells.num_z, J.data());
            for (i64 i = 0; i < J.totElems(); ++i) {
                state.J_cpu.data.data()[i] = Rad16ArrayHost::encode(J.data()[i]);
            }
        }
        state.background.have_J = true;
    }
    yakl::fence();
    int num_iter = 0;
    if (!in.att_int("num_iter", &num_iter)) {
        throw std::runtime_error(fmt::format("Unable to get num_iter from restart file {}", restart_path));
    }
    return num_iter;
}

/// Dump a (dense) snapshot. File name determined automatically (e.g. main
/// output dexrt_output.nc -> dexrt_output_snapshot.nc)
void save_snapshot(const ScState& state, int num_iter) {
    std::string name(state.config.output_path);
    const std::string ext(".nc");
    const std::string new_ext("_snapshot.nc");
    auto nc_pos = name.rfind(ext);
    if (nc_pos == std::string::npos) {
        // NOTE(cmo): Didn't find it, so just append
        name += new_ext;
    } else {
        name.replace(nc_pos, ext.size(), new_ext);
    }
    state.println("Saving snapshot to {}...", name);

    ScOutFile file(state.mpi_state, name, state.config.output.io);
    file.put_att("num_iter", i32(num_iter));
    file.put_att("program", std::string(state.cells.num_y > 1 ? "dexrt_sc (3d)" : "dexrt_sc (2d)"));
    file.put_att("output_format", std::string("full"));
    ScOutputWriter writer(file, state.slab, state.cells);
    writer.write_cells(state.pops, "pops", "level");
    if (state.config.conserve_charge) {
        writer.write_scalar(state.atmos.ne, "ne");
    }
    if (state.config.conserve_pressure) {
        writer.write_scalar(state.atmos.nh_tot, "nh_tot");
    }
    if (state.config.conserve_energy) {
        writer.write_scalar(state.atmos.temperature, "temperature");
        writer.write_scalar(state.atmos.pressure, "pressure");
    }
    if (state.config.snapshot_J && state.background.enabled && state.background.have_J) {
        // Preserve the encoded representation without a decoded full-volume copy.
        file.put_att("J_rad16", i32(HALF_RADIATION_STORAGE));
        writer.write_dense(
            state.J_cpu.data.reshape(state.J_cpu.extent(0), state.cells.num_z, state.cells.plane_size()),
            "J",
            "wavelength"
        );
    }
    if (state.prd.enabled()) {
        writer.write_cells(state.prd.rho.to_fp(), "prd_rho", "prd_row");
    }
}

/// Load (dense) populations from initial_pops_path (variable "pops").
void load_initial_pops(ScState* st, const std::string& initial_pops_path) {
    ScState& state(*st);
    state.println("Loading populations from {}.", initial_pops_path);
    ScInFile in(state.mpi_state, initial_pops_path);
    sc_read_cells(in, state, "pops", state.pops.extent(0)).deep_copy_to(state.pops);
    yakl::fence();
}
