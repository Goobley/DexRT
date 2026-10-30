#include "ScPrd.hpp"
#include "ScState.hpp"
#include "EmisOpac.hpp"
#include "Collisions.hpp"
#include "LoopUtils.hpp"
#include "Atmosphere.hpp"
#include "ScPreparedGeometry.hpp"
#include "ScFormal.hpp"
#include "ScGammaAccumulate.hpp"
#include <algorithm>

ScPrd sc_prd_init(const ScState& state, bool enabled) {
    ScPrd prd;
    const auto& adata = state.adata_host;
    std::vector<ScPrdLine> lines;
    yakl::Array<i32, 1, yakl::memHost> line_of_kr(
        "sc_prd_line_of_kr",
        std::max(i32(adata.lines.extent(0)), 1));
    line_of_kr = -1;
    i32 rows = 0;
    for (int kr = 0; kr < adata.lines.extent(0); ++kr) {
        const auto& l = adata.lines(kr);
        if (!enabled || l.type != LineProfileType::PrdVoigt || !has_gamma(adata.treatment(l.atom))) {
            continue;
        }
        line_of_kr(kr) = lines.size();
        lines.push_back(ScPrdLine{
            .kr = kr,
            .atom = l.atom,
            .i = l.i,
            .j = l.j,
            .blue = l.blue_idx,
            .red = l.red_idx,
            .row0 = rows,
            .lambda0 = l.lambda0,
            .mass = adata.mass(l.atom),
            .stim_ratio = l.Bji / l.Aji,
        });
        rows += l.red_idx - l.blue_idx;
        prd.reach_lo = lines.size() == 1 ? l.blue_idx : std::min(prd.reach_lo, l.blue_idx);
        prd.reach_hi = std::max(prd.reach_hi, l.red_idx);
    }
    prd.num_lines = lines.size();
    prd.num_rows = rows;
    yakl::Array<ScPrdLine, 1, yakl::memHost> lines_host("sc_prd_lines", std::max(prd.num_lines, 1));
    for (int p = 0; p < prd.num_lines; ++p) {
        lines_host(p) = lines[p];
    }
    prd.lines = lines_host.createDeviceCopy();
    prd.line_of_kr = line_of_kr.createDeviceCopy();
    prd.wavelength = state.adata.wavelength;
    const i64 num_cells = state.cells.num_cells();
    prd.rho = Rad16Array("sc_prd_rho", std::max(rows, 1), num_cells);
    prd.J_rest = Rad16Array("sc_prd_J_rest", std::max(rows, 1), num_cells);
    prd.C_out = Fp2d("sc_prd_C_out", std::max(prd.num_lines, 1), num_cells);
    prd.rho.fill(FP(1.0));
    prd.J_rest.fill(FP(0.0));
    prd.C_out = FP(0.0);
    yakl::fence();
    return prd;
}

void sc_prd_accumulate_J_rest(
    const ScState& state,
    const ScPrd& prd,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMedium& medium,
    const ScSolution& solution,
    i32 la_start,
    i32 wave_batch
) {
    using ConstantsFP::c;
    JasUnpack(medium, S);
    constexpr i32 halo = SC_MEDIUM_HALO;
    const auto& frames = batch.frames;
    const auto& dir_idx = batch.dir_idx;
    const auto& mu = dirs.mu;
    const auto& wmu = dirs.wmu;
    const i32 num_dirs = batch.num_dirs;
    const auto flat_atmos = flatten<const fp_t>(state.atmos);
    const ScCells cells = state.cells;
    const auto list = cells.list();
    const auto& wavelength = prd.wavelength;
    const auto& J_rest = prd.J_rest;
    const i32 la_end = la_start + wave_batch;
    const auto lines = prd.lines.createHostCopy();
    const auto& wl_h = state.adata_host.wavelength;
    for (int p = 0; p < prd.num_lines; ++p) {
        const ScPrdLine l = lines(p);
        // NOTE(cmo): The line's lab knots in this batch, and the comoving knots
        // whose images (shifted by at most max_speed lambda0 / c) they can
        // bracket; the rest add nothing. One thread per (comoving knot, cell),
        // so neighbouring threads read the same lab rows.
        const i32 lo = std::max(la_start, l.blue);
        const i32 hi = std::min(la_end, l.red);
        if (lo >= hi) {
            continue;
        }
        const fp_t max_shift = prd.max_speed * l.lambda0 / c;
        const auto bracket_h = [&](fp_t x) {
            return i32(std::upper_bound(&wl_h(l.blue), &wl_h(l.blue) + (l.red - l.blue), x) - &wl_h(l.blue)) +
                   l.blue - 1;
        };
        const i32 k_start = (lo == l.blue) ? l.blue : std::max(bracket_h(wl_h(lo - 1) - max_shift), l.blue);
        const i32 k_end = (hi == l.red) ? l.red : std::min(bracket_h(wl_h(hi) + max_shift) + 2, l.red);
        dex_parallel_for(
            "SC PRD J*",
            FlatLoop<2>(k_end - k_start, list.num()),
            YAKL_LAMBDA(i32 kk, i64 i) {
                JasUse(halo);
                const i32 k = k_start + kk;
                const i64 ks = list(i);
                const Coord3 coord = cells.coord(ks);
                const fp_t vx = flat_atmos.vx(ks);
                const fp_t vy = flat_atmos.vy(ks);
                const fp_t vz = flat_atmos.vz(ks);
                fp_t sum = FP(0.0);
                bool any = false;
                for (int d = 0; d < num_dirs; ++d) {
                    const i32 dir = dir_idx(d);
                    const fp_t vel = vx * mu(dir, 0) + vy * mu(dir, 1) + vz * mu(dir, 2);
                    const fp_t shift = vel * l.lambda0 / c;
                    // NOTE(cmo): Comoving knot k sits at lab wavelength x
                    // between the line's lab knots b, b + 1 (constant beyond
                    // its ends, as rho_lab: the knots outside don't see the
                    // line). The lab and comoving knots coincide, so b lies in
                    // [blue - 1, k] if shift >= 0, else [k, red - 1]: only that
                    // half is searched.
                    const fp_t x = wavelength(k) - shift;
                    const bool up = shift >= FP(0.0);
                    const i32 bd =
                        sc_bracket(wavelength, up ? l.blue - 1 : k, up ? k - l.blue + 2 : l.red - k, x);
                    i32 knot[2] = {bd, bd + 1};
                    fp_t weight[2] = {FP(0.0), FP(0.0)};
                    if (bd < l.blue) {
                        knot[0] = l.blue;
                        weight[0] = FP(1.0);
                    } else if (bd >= l.red - 1) {
                        knot[0] = l.red - 1;
                        weight[0] = FP(1.0);
                    } else {
                        const fp_t t = (x - wavelength(bd)) / (wavelength(bd + 1) - wavelength(bd));
                        weight[0] = FP(1.0) - t;
                        weight[1] = t;
                    }
                    const ScFrame f = frames(d);
                    const i64 idx = sc_canonical_idx(f, coord.x, coord.y, coord.z);
                    const i64 mat = idx + i64(halo) * sc_plane_size(f);
                    for (int n = 0; n < 2; ++n) {
                        if (weight[n] != FP(0.0) && knot[n] >= la_start && knot[n] < la_end) {
                            const i32 wave = knot[n] - la_start;
                            sum += wmu(dir) * weight[n] * solution.intensity(d, wave, idx, S(d, wave, mat));
                            any = true;
                        }
                    }
                }
                // NOTE(cmo): Summed over the batch's directions, then stored once.
                if (any) {
                    J_rest.add(l.row0 + k - l.blue, ks, sum);
                }
            });
    }
    Kokkos::fence();
}

void sc_prd_capture_collisions(const ScState& state, const ScPrd& prd) {
    const auto lines = prd.lines.createHostCopy();
    const auto& atmos = state.atmos;
    const auto& nh_lte = state.nh_lte;
    const auto& C_out = prd.C_out;
    const auto cells = cell_list(state);
    for (int p = 0; p < prd.num_lines; ++p) {
        const ScPrdLine l = lines(p);
        const auto atom = extract_atom(state.adata, state.adata_host, l.atom);
        const int num_level = state.adata_host.num_level(l.atom);
        size_t scratch_size = ScratchView<fp_t**>::shmem_size(num_level, num_level);
        scratch_size += ScratchView<fp_t*>::shmem_size(num_level);
        FlatLoop<2> nxn_loop(num_level, num_level);
        Kokkos::parallel_for(
            "SC PRD collisions",
            TeamPolicy(cells.num(), dex_team_size(std::min(square(num_level), 128)))
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const KTeam& team) {
                const i64 ks = cells(team.league_rank());
                ScratchView<fp_t**> Ck(team.team_scratch(0), num_level, num_level);
                ScratchView<fp_t*> n_star(team.team_scratch(0), num_level);
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nxn_loop.num_iter), [&](const int x) {
                    const auto args = nxn_loop.unpack(x);
                    Ck(args[0], args[1]) = FP(0.0);
                });
                team.team_barrier();
                scatter_add_collisions(
                    team,
                    atom,
                    collision_atmos(atmos, ks),
                    nh_lte,
                    n_star.data(),
                    Ck
                );
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    fp_t out = FP(0.0);
                    for (int k = 0; k < num_level; ++k) {
                        out += (k == l.j) ? FP(0.0) : Ck(k, l.j);
                    }
                    C_out(p, ks) = out;
                });
            });
    }
    Kokkos::fence();
}

fp_t sc_prd_redistribute(const ScState& state, const ScPrd& prd) {
    using namespace ScPrdGii;
    if (!prd.enabled()) {
        return FP(0.0);
    }
    if (!state.rate_diag.radiative_rates.initialized()) {
        throw std::runtime_error("SC PRD needs the radiative rates (rate_diag)");
    }
    const auto& R = state.rate_diag.radiative_rates;
    JasUnpack(state, adata, pops, phi);
    const auto flat_atmos = flatten<const fp_t>(state.atmos);
    const auto& wavelength = prd.wavelength;
    const auto& wavelength_bin = adata.wavelength_bin;
    const auto& rho = prd.rho;
    const auto& J_rest = prd.J_rest;
    const auto& C_out = prd.C_out;
    const i32 num_lines = prd.num_lines;

    fp_t max_change = FP(0.0);
    const auto cells = cell_list(state);
    dex_parallel_reduce(
        "SC PRD redistribute",
        FlatLoop<1>(cells.num()),
        KOKKOS_LAMBDA(i64 i, fp_t & running_max) {
            const i64 ks = cells(i);
            using ConstantsFP::sqrt_pi;
            const fp_t T = flat_atmos.temperature(ks);
            const fp_t vturb = flat_atmos.vturb(ks);
            const fp_t ne = flat_atmos.ne(ks);
            const fp_t nh0 = flat_atmos.nh0(ks);
            for (int p = 0; p < num_lines; ++p) {
                const ScPrdLine l = prd.lines(p);
                const auto& line = adata.lines(l.kr);
                const i32 offset = adata.level_start(l.atom);
                const fp_t ni = pops(offset + l.i, ks);
                const fp_t nj = pops(offset + l.j, ks);
                // NOTE(cmo): Radiative depopulation of j: down where it's the
                // upper level, up where it's the lower.
                fp_t P = C_out(p, ks);
                const int t0 = trans_start(adata, l.atom);
                const int num_trans = adata.num_line(l.atom) + adata.num_cont(l.atom);
                for (int t = 0; t < num_trans; ++t) {
                    const TransitionLevels lv = transition_levels(adata, l.atom, t);
                    if (lv.j == l.j) {
                        P += fp_t(R(t0 + t, PRECON_R_DOWN, ks));
                    } else if (lv.i == l.j) {
                        P += fp_t(R(t0 + t, PRECON_R_UP, ks));
                    }
                }
                const fp_t Q = sc_prd_elastic_rate(
                    line,
                    adata.broadening,
                    T,
                    ne,
                    nh0
                );
                const fp_t dop_width = doppler_width(l.lambda0, l.mass, T, vturb);
                const fp_t a = damping_from_gamma(
                    gamma_from_broadening(
                        line,
                        adata.broadening,
                        T,
                        ne,
                        nh0
                    ),
                    l.lambda0,
                    dop_width
                );
                const fp_t inv_dop = FP(1.0) / dop_width;
                const auto q_of = [&](i32 k) { return (wavelength(k) - l.lambda0) * inv_dop; };
                // NOTE(cmo): Jbar from J* and the static (comoving) profile.
                fp_t J_bar = FP(0.0);
                fp_t phi_sum = FP(0.0);
                for (int k = l.blue; k < l.red; ++k) {
                    const fp_t wphi_k = phi(a, std::abs(q_of(k))) * wavelength_bin(k);
                    J_bar += J_rest(l.row0 + k - l.blue, ks) * wphi_k;
                    phi_sum += wphi_k;
                }
                J_bar /= phi_sum;
                // NOTE(cmo): Impact limit: the coherence fraction gamma = P / (P + Q).
                const fp_t coherence = P / (P + Q);
                const fp_t prefactor =
                    (P > FP(0.0) && nj > FP(0.0)) ? coherence * ni * line.Bij / (nj * P) : FP(0.0);

                for (int e = l.blue; e < l.red; ++e) {
                    const fp_t q_emit = q_of(e);
                    fp_t q0, qN;
                    range(q_emit, &q0, &qN);
                    const i32 num_fine = i32((qN - q0) / dq) + 1;
                    // NOTE(cmo): J* on the fine absorption grid (linear in the
                    // line's knots, constant beyond), weighted by g_II with
                    // Lightweaver's end-corrected trapezoid weights, normalised.
                    i32 b = prd.bracket(l.blue, l.red, l.lambda0 + q0 * dop_width);
                    fp_t g_norm = FP(0.0);
                    fp_t scat = FP(0.0);
                    for (int n = 0; n < num_fine; ++n) {
                        const fp_t q_abs = q0 + n * dq;
                        const fp_t x = l.lambda0 + q_abs * dop_width;
                        while (b + 1 < l.red && wavelength(b + 1) <= x) {
                            ++b;
                        }
                        fp_t J;
                        if (b < l.blue) {
                            J = J_rest(l.row0, ks);
                        } else if (b >= l.red - 1) {
                            J = J_rest(l.row0 + l.red - 1 - l.blue, ks);
                        } else {
                            const fp_t t = (x - wavelength(b)) / (wavelength(b + 1) - wavelength(b));
                            const i32 r = l.row0 + b - l.blue;
                            J = (FP(1.0) - t) * J_rest(r, ks) + t * J_rest(r + 1, ks);
                        }
                        fp_t w = FP(1.0);
                        if (n == 0 || n == num_fine - 1) {
                            w = FP(5.0) / FP(12.0);
                        } else if (n == 1 || n == num_fine - 2) {
                            w = FP(13.0) / FP(12.0);
                        }
                        const fp_t g = gii(a, q_emit, q_abs) * w;
                        g_norm += g;
                        scat += J * g;
                    }
                    const i32 r = l.row0 + e - l.blue;
                    const fp_t updated = FP(1.0) + prefactor * (scat / g_norm - J_bar);
                    const fp_t old = rho(r, ks);
                    running_max =
                        std::max(running_max, std::abs(updated - old) / std::max(std::abs(old), FP(1e-30)));
                    rho.set(r, ks, updated);
                }
            }
        },
        Kokkos::Max<fp_t>(max_change));
    return max_change;
}
