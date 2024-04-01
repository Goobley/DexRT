#include "CrtafParser.hpp"
#include "catch_amalgamated.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Basic Parsing", "[crtaf_parsing]") {

    ModelAtom<double> model = parse_crtaf_model<double>("../test_atom.yaml");

    REQUIRE(model.element.symbol == "H");
    REQUIRE(model.levels[0].energy == 0.0);
    REQUIRE_THAT(model.levels[1].energy, WithinRel((10.198718355384056), 1e-5));
    REQUIRE(model.levels.size() == 4);
    REQUIRE(model.levels[2].g == 18);
    REQUIRE(model.levels[3].stage == 2);
    REQUIRE(model.levels[0].key == "i_1");
    REQUIRE(model.levels[3].key == "ii_1");

    REQUIRE(model.lines[0].type == LineProfileType::PrdVoigt);
    REQUIRE(model.lines[0].j == 1);
    REQUIRE(model.lines[0].i == 0);
    REQUIRE_THAT(model.lines[0].f, WithinRel(0.4162, 1e-6));
    REQUIRE_THAT(model.lines[0].lambda0, WithinRel(121.56841096386108, 1e-6));
    REQUIRE_THAT(model.lines[0].g_natural, WithinRel(470000000.0, 1e-6));
    REQUIRE(model.lines[0].wavelength.size() == 101);
    REQUIRE_THAT(model.lines[0].wavelength[0], WithinRel(model.lines[0].lambda0 - 0.01, 1e-6));
    REQUIRE_THAT(model.lines[0].wavelength.back(), WithinRel(model.lines[0].lambda0 + 0.01, 1e-6));

    REQUIRE(model.lines[1].type == LineProfileType::Voigt);
    REQUIRE(model.lines[1].j == 2);
    REQUIRE(model.lines[1].i == 0);
    REQUIRE_THAT(model.lines[1].Aji, WithinRel(55719499.00510443, 1e-6));
    REQUIRE_THAT(model.lines[1].Bji, WithinRel(151357509382.88724, 1e-6));
    REQUIRE_THAT(model.lines[1].Bji_wavelength, WithinRel(5.311929083369043e-12, 1e-6));
    REQUIRE_THAT(model.lines[1].Bij, WithinRel(1362217584445.985, 1e-6));
    REQUIRE_THAT(model.lines[1].Bij_wavelength, WithinRel(4.780736175032139e-11, 1e-6));
    REQUIRE_THAT(model.lines[1].g_natural, WithinRel(99800000.0, 1e-6));
    REQUIRE(model.lines[1].broadening.size() == 3);
    REQUIRE_THAT(model.lines[1].broadening[0].scaling, WithinRel(3.0937539490410914e-15, 1e-6));
    REQUIRE_THAT(model.lines[1].broadening[0].temperature_exponent, WithinRel(0.3, 1e-6));
    REQUIRE_THAT(model.lines[1].broadening[0].hydrogen_exponent, WithinRel(1.0, 1e-6));
    REQUIRE_THAT(model.lines[1].broadening[2].scaling, WithinRel(0.002563539605329271, 1e-6));
    REQUIRE_THAT(model.lines[1].broadening[2].temperature_exponent, WithinRel(0.0, 1e-6));
    REQUIRE_THAT(model.lines[1].broadening[2].hydrogen_exponent, WithinRel(0.0, 1e-6));
    REQUIRE_THAT(model.lines[1].broadening[2].electron_exponent, WithinRel(2.0 / 3.0, 1e-6));
    REQUIRE(model.lines[1].wavelength.size() == 39);
    REQUIRE_THAT(model.lines[1].wavelength[19], WithinRel(model.lines[1].lambda0, 1e-10));
    REQUIRE_THAT(model.lines[1].wavelength[32], WithinRel(model.lines[1].lambda0 + 0.020768013164324042, 1e-6));
    
    REQUIRE(model.lines[2].type == LineProfileType::Voigt);
    REQUIRE(model.lines[2].j == 2);
    REQUIRE(model.lines[2].i == 1);
    REQUIRE_THAT(model.lines[2].f, WithinRel(0.6407, 1e-6));
    REQUIRE_THAT(model.lines[2].Bji_wavelength, WithinRel(4.511593806866219e-08, 1e-6));

    REQUIRE(model.continua.size() == 3);
    REQUIRE(model.continua[1].wavelength.size() == 20);
    REQUIRE_THAT(model.continua[1].sigma[0], WithinRel(2.57898033746484e-23, 1e-6));
    REQUIRE_THAT(model.continua[1].wavelength[19], WithinRel(364.705201845941, 1e-6));

    REQUIRE(model.coll_rates.size() == 11);
    REQUIRE(model.coll_rates[0].temperature.size() == 6);
    REQUIRE_THAT(model.coll_rates[0].temperature[3], WithinRel(10000.0, 1e-6));
    REQUIRE_THAT(model.coll_rates[0].data[3], WithinRel(3.365e-16, 1e-6));

    REQUIRE(model.coll_rates[0].type == CollRateType::CE);
    REQUIRE(model.coll_rates[1].type == CollRateType::Omega);
    REQUIRE(model.coll_rates[2].type == CollRateType::CE);
    REQUIRE(model.coll_rates[3].type == CollRateType::CP);
    REQUIRE(model.coll_rates[4].type == CollRateType::CE);
    REQUIRE(model.coll_rates[5].type == CollRateType::CH);
    REQUIRE(model.coll_rates[6].type == CollRateType::CI);
    REQUIRE(model.coll_rates[7].type == CollRateType::ChargeExcH);
    REQUIRE(model.coll_rates[8].type == CollRateType::CI);
    REQUIRE(model.coll_rates[9].type == CollRateType::ChargeExcP);
    REQUIRE(model.coll_rates[10].type == CollRateType::CI);
}