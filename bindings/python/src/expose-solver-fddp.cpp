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
      .def(bp::init<const context::TrajOptProblem &>(
          bp::args("self", "problem")));

  bp::class_<SolverType, boost::noncopyable>(
      "SolverFDDP", "An implementation of the FDDP solver from Crocoddyl.",
      bp::init<Scalar, VerboseLevel, Scalar, std::size_t>(
          (bp::arg("self"), bp::arg("tol"),
           bp::arg("verbose") = VerboseLevel::QUIET, bp::arg("reg_init") = 1e-9,
           bp::arg("max_iters") = 1000)))
      .def_readwrite("reg_min", &SolverType::reg_min_)
      .def_readwrite("reg_max", &SolverType::reg_max_)
      .def_readwrite("xreg", &SolverType::xreg_)
      .def_readwrite("ureg", &SolverType::ureg_)
      .def(SolverVisitor<SolverType>())
      .def("run", &SolverType::run,
           (bp::arg("self"), bp::arg("problem"), bp::arg("xs_init"),
            bp::arg("us_init")));
}

} // namespace python
} // namespace aligator
