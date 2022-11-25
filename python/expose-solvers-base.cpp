#include "proxddp/python/fwd.hpp"

#include "proxddp/core/workspace.hpp"
#include "proxddp/core/results.hpp"

namespace proxddp {
namespace python {

/* fwd declarations */

void exposeFDDP();
void exposeProxDDP();

/// Expose base solver elements
void exposeBase() {
  using context::Scalar;

  using QParams = proxddp::internal::q_storage<Scalar>;
  using VParams = proxddp::internal::value_storage<Scalar>;
  bp::class_<QParams>(
      "QParams", "Q-function parameters.",
      bp::init<int, int, int>(bp::args("self", "ndx", "nu", "ndy")))
      .def_readonly("grad_", &QParams::grad_)
      .def_readonly("hess_", &QParams::hess_)
      .add_property(
          "Qx", bp::make_getter(&QParams::Qx,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Qu", bp::make_getter(&QParams::Qu,
                                bp::return_value_policy<bp::return_by_value>()))
      .def(PrintableVisitor<QParams>());

  bp::class_<VParams>("VParams", "Value function parameters.", bp::no_init)
      .def_readonly("Vx", &VParams::Vx_)
      .def_readonly("Vxx", &VParams::Vxx_)
      .def(PrintableVisitor<VParams>());

  StdVectorPythonVisitor<std::vector<QParams>, true>::expose("StdVec_QParams");
  StdVectorPythonVisitor<std::vector<VParams>, true>::expose("StdVec_VParams");

  using WorkspaceBase = WorkspaceBaseTpl<Scalar>;
  bp::class_<WorkspaceBase>("WorkspaceBase", "Base workspace struct.",
                            bp::no_init)
      .def_readonly("nsteps", &WorkspaceBase::nsteps)
      .def_readonly("problem_data", &WorkspaceBase::problem_data)
      .def_readonly("trial_xs", &WorkspaceBase::trial_xs)
      .def_readonly("trial_us", &WorkspaceBase::trial_us)
      .def_readonly("dyn_slacks", &WorkspaceBase::dyn_slacks,
                    "Expose dynamics' slack variables (e.g. feasibility gaps).")
      .def_readonly("value_params", &WorkspaceBase::value_params)
      .def_readonly("q_params", &WorkspaceBase::q_params);

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
      .add_property("ctrl_feedbacks", &ResultsBase::getCtrlFeedbacks,
                    "Get the control feedback matrices.")
      .add_property("ctrl_feedforwards", &ResultsBase::getCtrlFeedforwards,
                    "Get the control feedforward gains.")
      .def(PrintableVisitor<ResultsBase>());
}

void exposeSolvers() {
  exposeBase();
  exposeFDDP();
  exposeProxDDP();
}

} // namespace python
} // namespace proxddp
