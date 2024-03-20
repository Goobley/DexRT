#include "CrtafParser.hpp"
#include "catch_amalgamated.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("Basic Parsing", "[crtaf_parsing]") {

    ModelAtom model = parse_crtaf_model("../test_atom.yaml");

    REQUIRE(model.element.symbol == "H");
    REQUIRE(model.levels[0].energy == 0.0);
    REQUIRE_THAT(model.levels[1].energy, WithinRel((10.198718355384056), 1e-5));
}