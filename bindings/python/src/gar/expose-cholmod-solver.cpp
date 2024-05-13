#include "aligator/python/fwd.hpp"
#include "aligator/gar/cholmod-solver.hpp"

namespace aligator::python {

using lqr_t = gar::LQRProblemTpl<context::Scalar>;
using cholmod_solver_t = gar::CholmodLqSolver<context::Scalar>;

void exposeCholmodSolver() {
  bp::class_<cholmod_solver_t, boost::noncopyable>("CholmodLqSolver",
                                                   bp::no_init)
      .def(bp::init<const lqr_t &, uint>(
          ("self"_a, "problem", "numRefinementSteps"_a = 1)))
      .def_readonly("kktMatrix", &cholmod_solver_t::kktMatrix)
      .def_readonly("kktRhs", &cholmod_solver_t::kktRhs)
      .def("backward", &cholmod_solver_t::backward, ("self"_a, "mudyn", "mueq"))
      .def("forward", &cholmod_solver_t::forward,
           ("self"_a, "xs", "us", "vs", "lbdas"))
      .add_property("sparse_residual", &cholmod_solver_t::computeSparseResidual,
                    "Sparse problem residual.")
      .def_readonly("cholmod", &cholmod_solver_t::cholmod)
      .def_readwrite("numRefinementSteps",
                     &cholmod_solver_t::numRefinementSteps);
}

} // namespace aligator::python
