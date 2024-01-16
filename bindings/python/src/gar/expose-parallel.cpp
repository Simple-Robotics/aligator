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
  bp::class_<parallel_solver_t, boost::noncopyable>("ParallelRiccatiSolver",
                                                    bp::no_init)
      .def(bp::init<const lqr_t &, uint>(("self"_a, "problem", "num_threads")))
      .def_readonly("datas", &parallel_solver_t::datas)
      .def("backward", &parallel_solver_t::backward,
           ("self"_a, "mudyn", "mueq"))
      .def("forward", &parallel_solver_t::forward,
           ("self"_a, "xs", "us", "vs", "lbdas"));
}

} // namespace python
} // namespace aligator
