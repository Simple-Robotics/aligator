#include "aligator/python/fwd.hpp"

#include "aligator/solvers/fddp/solver-fddp.hpp"

namespace aligator::python {
using context::Scalar;
using QParams = QFunctionTpl<Scalar>;
using VParams = ValueFunctionTpl<Scalar>;
} // namespace aligator::python

#if EIGENPY_VERSION_AT_MOST(3, 2, 0)
namespace eigenpy {

template <>
struct has_operator_equal<::aligator::python::QParams> : boost::false_type {};
template <>
struct has_operator_equal<::aligator::python::VParams> : boost::false_type {};

} // namespace eigenpy
#endif

namespace aligator {
namespace python {

void exposeFDDP() {
  using context::Manifold;
  using context::SolverFDDP;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;

  bp::class_<QParams>("QParams", "Q-function parameters.",
                      bp::init<int, int, int>(("self"_a, "ndx", "nu", "ndy")))
      .add_property("ntot", &QParams::ntot)
      .def_readonly("grad", &QParams::grad_)
      .def_readonly("hess", &QParams::hess_)
      .add_property(
          "Qx", bp::make_getter(&QParams::Qx,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Qu", bp::make_getter(&QParams::Qu,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property("Qxx", bp::make_getter(
                               &QParams::Qxx,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Qxu", bp::make_getter(
                               &QParams::Qxu,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Qxy", bp::make_getter(
                               &QParams::Qxy,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Quu", bp::make_getter(
                               &QParams::Quu,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Quy", bp::make_getter(
                               &QParams::Quy,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Qyy", bp::make_getter(
                               &QParams::Qyy,
                               bp::return_value_policy<bp::return_by_value>()))
      .def(PrintableVisitor<QParams>());

  bp::class_<VParams>("VParams", "Value function parameters.", bp::no_init)
      .def_readonly("Vx", &VParams::Vx_)
      .def_readonly("Vxx", &VParams::Vxx_)
      .def(PrintableVisitor<VParams>());

  StdVectorPythonVisitor<std::vector<QParams>>::expose("StdVec_QParams");
  StdVectorPythonVisitor<std::vector<VParams>>::expose("StdVec_VParams");

  bp::class_<Workspace, bp::bases<Workspace::Base>>("WorkspaceFDDP",
                                                    bp::no_init)
      .def_readonly("dxs", &Workspace::dxs)
      .def_readonly("dus", &Workspace::dus)
      .def_readonly("value_params", &Workspace::value_params)
      .def_readonly("q_params", &Workspace::q_params)
      .def_readonly("d1", &Workspace::d1_)
      .def_readonly("d2", &Workspace::d2_);

  bp::class_<Results, bp::bases<Results::Base>>("ResultsFDDP", bp::no_init)
      .def(bp::init<const context::TrajOptProblem &>(("self"_a, "problem")));

  bp::class_<SolverFDDP, boost::noncopyable>(
      "SolverFDDP", "An implementation of the FDDP solver from Crocoddyl.",
      bp::init<Scalar, VerboseLevel, Scalar, std::size_t>(
          ("self"_a, "tol", "verbose"_a = VerboseLevel::QUIET,
           "reg_init"_a = 1e-9, "max_iters"_a = 1000)))
      .def_readwrite("reg_min", &SolverFDDP::reg_min_)
      .def_readwrite("reg_max", &SolverFDDP::reg_max_)
      .def_readwrite("xreg", &SolverFDDP::xreg_)
      .def_readwrite("ureg", &SolverFDDP::ureg_)
      .def(SolverVisitor<SolverFDDP>())
      .def("run", &SolverFDDP::run,
           ("self"_a, "problem", "xs_init", "us_init"));
}

} // namespace python
} // namespace aligator
