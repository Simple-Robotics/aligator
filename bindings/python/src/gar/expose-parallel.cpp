/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/gar-visitors.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/gar/parallel-solver.hpp"

namespace aligator {
namespace python {

using namespace gar;
using context::Scalar;
using knot_t = LQRKnotTpl<context::Scalar>;
using lqr_t = LQRProblemTpl<context::Scalar>;
using parallel_solver_t = gar::ParallelRiccatiSolver<Scalar>;

void exposeParallelSolver() {
}

} // namespace python
} // namespace aligator
