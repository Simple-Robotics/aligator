/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/blk-matrix.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/gar/parallel-solver.hpp"

namespace aligator {
namespace python {

using namespace gar;
using context::Scalar;
using riccati_base_t = RiccatiSolverBase<Scalar>;
using knot_t = LqrKnotTpl<context::Scalar>;
using lqr_t = LQRProblemTpl<context::Scalar>;

void exposeParallelSolver() {
#ifdef ALIGATOR_MULTITHREADING
  using parallel_solver_t = gar::ParallelRiccatiSolver<Scalar>;
  bp::class_<parallel_solver_t, bp::bases<riccati_base_t>, boost::noncopyable>(
      "ParallelRiccatiSolver", bp::no_init)
      .def(bp::init<lqr_t &, uint>(("self"_a, "problem", "num_threads")))
      .def_readonly("datas", &parallel_solver_t::datas);
#endif
}

} // namespace python
} // namespace aligator
