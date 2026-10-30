#include "ScFormal.hpp"
#include "ScShortSweep.hpp"

namespace {
/// Select the compiled interpolation variant before entering the sweep kernel.
template <ScFormalSolver Solver, class Run> void sc_dispatch_interp(ScInterp interp, const Run& run) {
    switch (interp) {
    case ScInterp::Linear:
        return run.template operator()<Solver, ScInterp::Linear>();
    case ScInterp::VanLeer:
        return run.template operator()<Solver, ScInterp::VanLeer>();
    case ScInterp::MonotonizedCentral:
        return run.template operator()<Solver, ScInterp::MonotonizedCentral>();
    }
    throw std::runtime_error("Unhandled SC interpolation");
}

/// Select one compiled numerical variant before entering the device kernel.
template <class Run> void sc_dispatch_short(ScFormalSolver solver, ScInterp interp, const Run& run) {
    switch (solver) {
    case ScFormalSolver::PiecewiseConstant:
        return sc_dispatch_interp<ScFormalSolver::PiecewiseConstant>(interp, run);
    case ScFormalSolver::PiecewiseLinear:
        return sc_dispatch_interp<ScFormalSolver::PiecewiseLinear>(interp, run);
    case ScFormalSolver::PiecewiseLinearEmissivity:
        return sc_dispatch_interp<ScFormalSolver::PiecewiseLinearEmissivity>(interp, run);
    }
    throw std::runtime_error("Unhandled SC formal solver");
}

} // namespace

void sc_solve_short(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    ScFormalSolver solver,
    ScInterp interp
) {
    sc_dispatch_short(solver, interp, [&]<ScFormalSolver Solver, ScInterp Interp>() {
        sc_dispatch_bool(geometry.grid.dim == 3, [&]<bool ThreeD>() {
            sc_dispatch_bool(task.loss.enabled(), [&]<bool Loss>() {
                sc_short_sweep<ThreeD ? 3 : 2, Solver, Interp, Loss>(geometry, work, task);
            });
        });
    });
}
