#include "proxddp/python/fwd.hpp"

#include "proxddp/core/workspace-base.hpp"
#include "proxddp/core/results.hpp"

namespace proxddp {
namespace python {

/* fwd declarations */

void exposeFDDP();
void exposeProxDDP();

/// Expose base solver elements
void exposeSolverCommon() {
  using context::Scalar;

  using QParams = proxddp::QFunctionTpl<Scalar>;
  using VParams = proxddp::ValueFunctionTpl<Scalar>;
  bp::class_<QParams>(
      "QParams", "Q-function parameters.",
      bp::init<int, int, int>(bp::args("self", "ndx", "nu", "ndy")))
      .add_property("ntot", &QParams::ntot)
      .def_readonly("grad", &QParams::grad_)
      .def_readonly("hess", &QParams::hess_)
      .add_property(
          "Qx", bp::make_getter(&QParams::Qx,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Qu", bp::make_getter(&QParams::Qu,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Qy", bp::make_getter(&QParams::Qy,
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
      .def(bp::self == bp::self)
      .def(PrintableVisitor<QParams>());

  bp::class_<VParams>("VParams", "Value function parameters.", bp::no_init)
      .def_readonly("Vx", &VParams::Vx_)
      .def_readonly("Vxx", &VParams::Vxx_)
      .def(PrintableVisitor<VParams>());

  StdVectorPythonVisitor<std::vector<QParams>, true>::expose("StdVec_QParams");
  StdVectorPythonVisitor<std::vector<VParams>, true>::expose("StdVec_VParams");

  using WorkspaceBase = WorkspaceBaseTpl<Scalar>;
  bp::class_<WorkspaceBase, boost::noncopyable>(
      "WorkspaceBase", "Base workspace struct.", bp::no_init)
      .def_readonly("nsteps", &WorkspaceBase::nsteps)
      .def_readonly("problem_data", &WorkspaceBase::problem_data)
      .def_readonly("trial_xs", &WorkspaceBase::trial_xs)
      .def_readonly("trial_us", &WorkspaceBase::trial_us)
      .def_readonly("dyn_slacks", &WorkspaceBase::dyn_slacks,
                    "Expose dynamics' slack variables (e.g. feasibility gaps).")
      .def_readonly("value_params", &WorkspaceBase::value_params)
      .def_readonly("q_params", &WorkspaceBase::q_params)
      .def("cycleLeft", &WorkspaceBase::cycleLeft, bp::args("self"),
           "Cycle the workspace to the left: this will rotate all the data "
           "(states, controls, multipliers) forward by one rank.")
      .def("cycleAppend", &WorkspaceBase::cycleAppend, bp::args("self", "data"),
           "Insert a StageData object and cycle the "
           "workspace left (using `cycleLeft()`) and insert the allocated data "
           "(useful for MPC).");

  using ResultsBase = ResultsBaseTpl<Scalar>;
  bp::class_<ResultsBase>("ResultsBase", "Base results struct.", bp::no_init)
      .def_readonly("num_iters", &ResultsBase::num_iters,
                    "Number of solver iterations.")
      .def_readonly("conv", &ResultsBase::conv)
      .def_readonly("gains", &ResultsBase::gains_)
      .def_readonly("xs", &ResultsBase::xs)
      .def_readonly("us", &ResultsBase::us)
      .def_readonly("lams", &ResultsBase::lams)
      .def_readonly("primal_infeas", &ResultsBase::prim_infeas)
      .def_readonly("dual_infeas", &ResultsBase::dual_infeas)
      .def_readonly("traj_cost", &ResultsBase::traj_cost_, "Trajectory cost.")
      .def_readonly("merit_value", &ResultsBase::merit_value_,
                    "Merit function value.")
      .def("controlFeedbacks", &ResultsBase::getCtrlFeedbacks, bp::args("self"),
           "Get the control feedback matrices.")
      .def("controlFeedforwards", &ResultsBase::getCtrlFeedforwards,
           bp::args("self"), "Get the control feedforward gains.")
      .def(PrintableVisitor<ResultsBase>());
}

void exposeSolvers() {
  exposeSolverCommon();
  exposeFDDP();
  exposeProxDDP();
}

} // namespace python
} // namespace proxddp
