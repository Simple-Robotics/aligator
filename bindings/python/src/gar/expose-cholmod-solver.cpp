#include "aligator/python/fwd.hpp"
#include "aligator/gar/cholmod-solver.hpp"

namespace aligator::python {

using lqr_t = gar::LQRProblemTpl<context::Scalar>;
using cholmod_solver_t = gar::CholmodLqSolver<context::Scalar>;

void exposeCholmodSolver() {
  bp::class_<cholmod_solver_t, boost::noncopyable>("CholmodLqSolver",
                                                   bp::no_init)
      .def(bp::init<const lqr_t &>(("self"_a, "problem")))
      .def_readonly("kktMatrix", &cholmod_solver_t::kktMatrix);
}

} // namespace aligator::python
