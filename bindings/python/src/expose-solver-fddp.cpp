#include "aligator/python/fwd.hpp"

#include "aligator/solvers/fddp/solver-fddp.hpp"

namespace aligator {
namespace python {

void exposeFDDP() {
  using context::Manifold;
  using context::Scalar;
  using SolverType = SolverFDDP<Scalar>;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;

  bp::class_<Workspace, bp::bases<Workspace::Base>>("WorkspaceFDDP",
                                                    bp::no_init)
      .def_readonly("dxs", &Workspace::dxs)
      .def_readonly("dus", &Workspace::dus)
      .def_readonly("d1", &Workspace::d1_)
      .def_readonly("d2", &Workspace::d2_);

  bp::class_<Results, bp::bases<Results::Base>>("ResultsFDDP", bp::no_init)
      .def(bp::init<const context::TrajOptProblem &>(("self"_a, "problem")));

  bp::class_<SolverType, boost::noncopyable>(
      "SolverFDDP", "An implementation of the FDDP solver from Crocoddyl.",
      bp::init<Scalar, VerboseLevel, Scalar, std::size_t>(
          ("self"_a, "tol", "verbose"_a = VerboseLevel::QUIET,
           "reg_init"_a = 1e-9, "max_iters"_a = 1000)))
      .def_readwrite("reg_min", &SolverType::reg_min_)
      .def_readwrite("reg_max", &SolverType::reg_max_)
      .def_readwrite("xreg", &SolverType::xreg_)
      .def_readwrite("ureg", &SolverType::ureg_)
      .def(SolverVisitor<SolverType>())
      .def("run", &SolverType::run,
           ("self"_a, "problem", "xs_init", "us_init"));
}

} // namespace python
} // namespace aligator
