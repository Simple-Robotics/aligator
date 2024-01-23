#include "aligator/python/fwd.hpp"

#include "aligator/solvers/proxddp/results.hpp"
#include "aligator/core/workspace-base.hpp"
#include "aligator/core/value-function.hpp"

namespace aligator {
namespace python {

/* fwd declarations */

void exposeFDDP();
void exposeProxDDP();

/// Expose base solver elements
void exposeSolverCommon() {
  using context::Scalar;

  using WorkspaceBase = WorkspaceBaseTpl<Scalar>;
  bp::class_<WorkspaceBase, boost::noncopyable>(
      "WorkspaceBase", "Base workspace struct.", bp::no_init)
      .def_readonly("nsteps", &WorkspaceBase::nsteps)
      .def_readonly("problem_data", &WorkspaceBase::problem_data)
      .def_readonly("trial_xs", &WorkspaceBase::trial_xs)
      .def_readonly("trial_us", &WorkspaceBase::trial_us)
      .def_readonly("dyn_slacks", &WorkspaceBase::dyn_slacks,
                    "Expose dynamics' slack variables (e.g. feasibility gaps).")
      .def("cycleLeft", &WorkspaceBase::cycleLeft, "self"_a,
           "Cycle the workspace to the left: this will rotate all the data "
           "(states, controls, multipliers) forward by one rank.")
      .def("cycleAppend", &WorkspaceBase::cycleAppend, ("self"_a, "data"),
           "Insert a StageData object and cycle the workspace left (using "
           "`cycleLeft()`) and insert the allocated data (useful for MPC).");

  using ResultsBase = ResultsBaseTpl<Scalar>;
  bp::class_<ResultsBase>("ResultsBase", "Base results struct.", bp::no_init)
      .def_readonly("num_iters", &ResultsBase::num_iters,
                    "Number of solver iterations.")
      .def_readonly("conv", &ResultsBase::conv)
      .def_readonly("gains", &ResultsBase::gains_)
      .def_readonly("xs", &ResultsBase::xs)
      .def_readonly("us", &ResultsBase::us)
      .def_readonly("primal_infeas", &ResultsBase::prim_infeas)
      .def_readonly("dual_infeas", &ResultsBase::dual_infeas)
      .def_readonly("traj_cost", &ResultsBase::traj_cost_, "Trajectory cost.")
      .def_readonly("merit_value", &ResultsBase::merit_value_,
                    "Merit function value.")
      .def("controlFeedbacks", &ResultsBase::getCtrlFeedbacks, "self"_a,
           "Get the control feedback matrices.")
      .def("controlFeedforwards", &ResultsBase::getCtrlFeedforwards, "self"_a,
           "Get the control feedforward gains.")
      .def(PrintableVisitor<ResultsBase>());
}

void exposeSolvers() {
  exposeSolverCommon();
  exposeFDDP();
  exposeProxDDP();
}

} // namespace python
} // namespace aligator
