#include "CrtafParser.hpp"
#include "catch_amalgamated.hpp"

TEST_CASE("Basic Parsing", "[crtaf_parsing]") {

    ModelAtom model = parse_crtaf_model("../test_atom.yaml");

    REQUIRE(model.element.symbol == "H");
    REQUIRE(model.levels[0].energy == FP(0.0));
}