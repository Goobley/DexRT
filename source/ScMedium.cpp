#include "ScMedium.hpp"
#include "ScRayEntry.hpp"
#include "ScMediumCoefficients.hpp"
#include "ScPreparedGeometry.hpp"
#include "LoopUtils.hpp"
#include "Atmosphere.hpp"
#include <type_traits>

namespace {
/// Evaluate directional coefficients over active cells and explicitly zero inactive cells, including halos.
void sc_prepare_medium_atoms(
    const ScState& state,
    const ScMediumCoefficients& medium_coefficients,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const Fp3d& chi,
    const Fp3d& S,
    i32 la_start,
    i32 wave_batch
) {
    JasUnpack(state, atmos, adata, phi);
    const auto flatmos = flatten<const fp_t>(atmos);
    JasUnpack(batch, frames, dir_idx);
    JasUnpack(dirs, mu);
    JasUnpack(state, cells);
    JasUnpack(cells, runs, active);
    // NOTE(cmo): Halo rows included (local z in [-H, num_z + H)).
    {
        dex_parallel_for(
            "SC clear inactive medium",
            FlatLoop<4>(batch.num_dirs, wave_batch, cells.num_rows(), cells.plane_size()),
            YAKL_LAMBDA(i32 d, i32 wave, i32 row, i64 xy) {
                if (active(row, xy)) {
                    return;
                }
                const ScFrame f = frames(d);
                const i32 y = xy / f.num_s;
                const i32 x = xy - i64(y) * f.num_s;
                const i64 idx =
                    sc_canonical_idx(f, x, y, row - SC_MEDIUM_HALO) + i64(SC_MEDIUM_HALO) * sc_plane_size(f);
                chi(d, wave, idx) = FP(0.0);
                S(d, wave, idx) = FP(0.0);
            });
    }

    constexpr i32 G = SC_CELL_GROUP;
    dex_parallel_for(
        "SC prepare medium",
        FlatLoop<3>(batch.num_dirs, wave_batch, cells.num_runs_with_halo),
        YAKL_LAMBDA(i32 d, i32 wave, i64 r) {
            JasUse(G);
            const auto piece = runs(r);
            const i64 ks0 = cells.ks(piece.x0, piece.y, piece.z);
            const i32 n = piece.length();
            const ScFrame f = frames(d);
            const i64 idx0 =
                sc_canonical_idx(f, piece.x0, piece.y, piece.z) + i64(SC_MEDIUM_HALO) * sc_plane_size(f);
            const i32 la = la_start + wave;

            const i32 dir = dir_idx(d);
            const fp_t mu_x = mu(dir, 0);
            const fp_t mu_y = mu(dir, 1);
            const fp_t mu_z = mu(dir, 2);
            const fp_t* __restrict vx = &flatmos.vx(ks0);
            const fp_t* __restrict vy = &flatmos.vy(ks0);
            const fp_t* __restrict vz = &flatmos.vz(ks0);
            fp_t vel[G];
            SC_GROUP_SIMD
            for (int c = 0; c < n; ++c) {
                vel[c] = vx[c] * mu_x + vy[c] * mu_y + vz[c] * mu_z;
            }
            fp_t eta[G];
            fp_t chi_g[G];
            medium_coefficients.evaluate<G>(
                phi,
                ks0,
                wave,
                adata.wavelength(la),
                vel,
                eta,
                chi_g,
                n
            );
            fp_t* __restrict chi_p = &chi(d, wave, idx0);
            fp_t* __restrict S_p = &S(d, wave, idx0);
            // NOTE(cmo): x flips hoisted, for unit stride stores.
            const auto store = [&](auto step) {
                constexpr i64 st = decltype(step)::value;
                SC_GROUP_SIMD
                for (int c = 0; c < n; ++c) {
                    chi_p[st * c] = chi_g[c];
                    S_p[st * c] = sc_medium_from_eta(chi_g[c], eta[c]).S;
                }
            };
            if (f.flip_x) {
                store(std::integral_constant<i64, -1>{});
            } else {
                store(std::integral_constant<i64, 1>{});
            }
        });
    Kokkos::fence();
}

/// Map prescribed emissivity/opacity into canonical task storage, including halo rows.
void sc_prepare_medium_given(
    const ScState& state,
    const ScDirBatch& batch,
    const Fp3d& chi,
    const Fp3d& S,
    i32 la_start,
    i32 wave_batch
) {
    JasUnpack(state, given_state);
    JasUnpack(batch, frames);
    JasUnpack(given_state, emis, opac);
    const i32 num_rows = emis.extent(0);
    const i32 plane = emis.extent(1);

    // NOTE(cmo): Halo rows included.
    dex_parallel_for(
        "SC prepare medium (given)",
        FlatLoop<4>(batch.num_dirs, wave_batch, num_rows, plane),
        YAKL_LAMBDA(i32 d, i32 wave, i32 row, i32 xy) {
            const ScFrame f = frames(d);
            const i32 y = xy / f.num_s;
            const i32 x = xy - y * f.num_s;
            const i64 idx =
                sc_canonical_idx(f, x, y, row - SC_MEDIUM_HALO) + i64(SC_MEDIUM_HALO) * sc_plane_size(f);
            const i32 la = la_start + wave;
            const fp_t eta = emis(row, xy, la);
            const fp_t c = opac(row, xy, la);
            chi(d, wave, idx) = c;
            S(d, wave, idx) = sc_medium_from_eta(c, eta).S;
        });
    Kokkos::fence();
}

} // namespace
void sc_prepare_medium(
    const ScState& state,
    const ScDirectionSet& dirs,
    const ScDirBatch& batch,
    const ScMediumCoefficients* coefficients,
    const ScMedium& medium,
    i32 la,
    i32 width
) {
    JasUnpack(medium, chi, S);
    if (coefficients) {
        sc_prepare_medium_atoms(
            state,
            *coefficients,
            dirs,
            batch,
            chi,
            S,
            la,
            width
        );
    } else {
        sc_prepare_medium_given(
            state,
            batch,
            chi,
            S,
            la,
            width
        );
    }
}
