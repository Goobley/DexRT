#include "catch_amalgamated.hpp"
#include "Types.hpp"

/// Kokkos/YAKL lifetime for the standalone dexrt_sc_test executable.
class ScKokkosManager : public Catch::EventListenerBase {
    public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo&) override {
        Kokkos::initialize();
        yakl::init();
    }

    void testRunEnded(const Catch::TestRunStats&) override {
        yakl::finalize();
        Kokkos::finalize();
    }
};
CATCH_REGISTER_LISTENER(ScKokkosManager);
