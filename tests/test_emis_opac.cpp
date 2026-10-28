#include "catch_amalgamated.hpp"
#include "CrtafParser.hpp"
#include "EmisOpac.hpp"
#include "Utils.hpp"
#include "State.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE( "Test Emis Opac LTE CaII", "[emis_opac]" ) {
    {
        // Points [29, 30, 31] of FAL C from Lightweaver
        AtmosPointParams atmos[3] = {
            AtmosPointParams{
                FP(9358.0),
                FP(5.841142999999999e16),
                FP(7452.542),
                FP(1.1230805474824998e17),
                FP(0.0),
                FP(63145751664873.24)
            },
            AtmosPointParams{
                FP(9228.0),
                FP(6.141941999999999e+16),
                FP(7399.466),
                FP(1.1666706228599998e+17),
                FP(0.0),
                FP(89305991548028.25),
            },
            AtmosPointParams{
                FP(8988.0),
                FP(6.651615999999999e+16),
                FP(7268.274),
                FP(1.2817307512372998e+17),
                FP(0.0),
                FP(174405534535138.75)
            }
        };

        auto model = parse_crtaf_model<f64>("../test_CaII.yaml");
        // auto atom = to_comp_atom(model);
        auto atomic_data = to_atomic_data<fp_t, f64>({model});
        constexpr int num_wave = 8;
        const fp_t wavelengths[num_wave] = {
            fp_t(model.lines[0].lambda0),
            fp_t(model.lines[0].lambda0),
            fp_t(model.lines[model.lines.size()-1].lambda0 - FP(1.0)),
            fp_t(model.lines[model.lines.size()-1].lambda0 - FP(1.0)),
            fp_t(model.lines[model.lines.size()-1].lambda0),
            fp_t(model.lines[model.lines.size()-1].lambda0),
            FP(75.0),
            FP(75.0)
        };
        constexpr fp_t eta_expec[num_wave] = {FP(1.41178e-07), FP(2.38575e-07), FP(1.45433e-15), FP(1.11952e-15), FP(3.08896e-09), FP(4.04149e-09), FP(2.7571e-16), FP(2.76916e-16)};
        constexpr fp_t chi_expec[num_wave] = {FP(5.81712e-10), FP(1.09391e-09), FP(2.88559e-17), FP(2.15499e-17), FP(5.95996e-11), FP(8.03768e-11), FP(5.86328e-15), FP(1.02592e-14)};

        constexpr int atmos_idx[num_wave] = {1, 2, 1, 0, 0, 1, 1, 2};

        auto wavelength_grid = atomic_data.host.wavelength;
        yakl::Array<int, 1, yakl::memHost> wave_idxs("idxs", num_wave);
        for (int i = 0; i < wave_idxs.extent(0); ++i) {
            fp_t min_dist = FP(1.0);
            for (int la = 0; la < wavelength_grid.extent(0); ++la) {
                fp_t dist = std::abs(wavelength_grid(la) - wavelengths[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    wave_idxs(i) = la;
                }
            }
            CAPTURE(min_dist);
        }
        yakl::Array<int, 1, yakl::memDevice> las = wave_idxs.createDeviceCopy();

        VoigtProfile<fp_t> profile(
            VoigtProfile<fp_t>::Linspace{FP(0.0), FP(0.1), 1024},
            VoigtProfile<fp_t>::Linspace{FP(0.0), FP(1.5e3), 64 * 1024}
        ); // 256 MB

        Fp1d eta("eta", las.extent(0));
        Fp1d chi("chi", las.extent(0));
        Fp2d lte("lte", atomic_data.host.energy.extent(0), 3);
        Fp2d n_star_scratch("scratch", atomic_data.host.energy.extent(0), 3);

        const auto& adata = atomic_data.device;
        REQUIRE(atomic_data.host.num_level.extent(0) == 1);
        dex_parallel_for(
            "Compute Emis Opac",
            FlatLoop<1>(1),
            YAKL_LAMBDA (int _) {
                for (int i = 0; i < sizeof(atmos) / sizeof(atmos[0]); ++i) {
                    const auto& a = atmos[i];
                    lte_pops(
                        adata.energy,
                        adata.g,
                        adata.stage,
                        a.temperature,
                        a.ne,
                        adata.abundance(0)  * a.nhtot,
                        lte,
                        i
                    );
                }

                for (int i = 0; i < sizeof(atmos_idx) / sizeof(atmos_idx[0]); ++i) {
                    auto result = emis_opac<fp_t, yakl::memDevice>(
                        EmisOpacState<fp_t>{
                            .adata = adata,
                            .profile = profile,
                            .la = las(i),
                            .n = lte,
                            .n_star_scratch = n_star_scratch,
                            .k = atmos_idx[i],
                            .atmos = atmos[atmos_idx[i]]
                        }
                    );
                    eta(i) = result.eta;
                    chi(i) = result.chi;
                }
            }
        );
        yakl::fence();
        auto eta_h = eta.createHostCopy();
        auto chi_h = chi.createHostCopy();
        auto lte_h = lte.createHostCopy();

        // NOTE(cmo): The tolerance is relatively low here due to an accumulation of factors...
        // - difference in LTE pops
        // - potential single precision (but failure is identical in double \o/)
        // - post-calculation conversion + different constants in lw (also feed into LTE pops)
        for (int i = 0; i < eta_h.extent(0); ++i) {
            REQUIRE_THAT(eta_h(i), WithinRel(eta_expec[i], FP(1e-2)));
            REQUIRE_THAT(chi_h(i), WithinRel(chi_expec[i], FP(1e-2)));
        }

    }
}