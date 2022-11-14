/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"

namespace proxddp {
namespace python {
void exposeProblem() {
  using context::CostBase;
  using context::Manifold;
  using context::Scalar;
  using context::StageData;
  using context::StageModel;
  using context::TrajOptData;
  using context::TrajOptProblem;
  using InitCstrType = StateErrorResidualTpl<Scalar>;

  bp::class_<TrajOptProblem>(
      "TrajOptProblem", "Define a shooting problem.",
      bp::init<const context::VectorXs &,
               const std::vector<shared_ptr<StageModel>> &,
               const shared_ptr<CostBase> &>(
          bp::args("self", "x0", "stages", "term_cost")))
      .def(bp::init<const context::VectorXs &, const int,
                    const shared_ptr<context::Manifold> &,
                    const shared_ptr<CostBase> &>(
          bp::args("self", "x0", "nu", "space", "term_cost")))
      .def<void (TrajOptProblem::*)(const shared_ptr<StageModel> &)>(
          "addStage", &TrajOptProblem::addStage, bp::args("self", "new_stage"),
          "Add a stage to the problem.")
      .def(bp::init<InitCstrType, int, shared_ptr<CostBase>>(
          "Constructor adding the initial constraint explicitly.",
          bp::args("self", "init_constraint", "nu", "term_cost")))
      .def_readonly("stages", &TrajOptProblem::stages_,
                    "Stages of the shooting problem.")
      .def_readwrite("term_cost", &TrajOptProblem::term_cost_,
                     "Problem terminal cost.")
      .add_property("num_steps", &TrajOptProblem::numSteps,
                    "Number of stages in the problem.")
      .add_property("x0_init",
                    bp::make_function(&TrajOptProblem::getInitState,
                                      bp::return_internal_reference<>()),
                    &TrajOptProblem::setInitState, "Initial state.")
      .add_property("init_cstr", &TrajOptProblem::init_state_error,
                    "Get initial state constraint.")
      .def("setTerminalConstraint", &TrajOptProblem::setTerminalConstraint,
           bp::args("self", "constraint"), "Set terminal constraint.")
      .def("evaluate", &TrajOptProblem::evaluate,
           bp::args("self", "xs", "us", "prob_data"),
           "Evaluate the problem costs, dynamics, and constraints.")
      .def("computeDerivatives", &TrajOptProblem::computeDerivatives,
           bp::args("self", "xs", "us", "prob_data"),
           "Evaluate the problem derivatives. Call `evaluate()` first.")
      .def("replaceStageCircular", &TrajOptProblem::replaceStageCircular,
           bp::args("self", "model"),
           "Circularly replace the last stage in the problem, dropping the "
           "first stage.");

  bp::register_ptr_to_python<shared_ptr<TrajOptData>>();
  bp::class_<TrajOptData>(
      "TrajOptData", "Data struct for shooting problems.",
      bp::init<const TrajOptProblem &>(bp::args("self", "problem")))
      .def_readwrite("term_cost", &TrajOptData::term_cost_data,
                     "Terminal cost data.")
      .def_readwrite("term_constraint", &TrajOptData::term_cstr_data,
                     "Terminal constraint data.")
      .add_property(
          "stage_data",
          bp::make_getter(&TrajOptData::stage_data,
                          bp::return_value_policy<bp::return_by_value>()),
          "Data for each stage.");

  bp::def("computeTrajectoryCost", &computeTrajectoryCost<context::Scalar>,
          bp::args("problem", "data"),
          "Compute the cost of the trajectory. NOTE: problem.evaluate() must "
          "be called beforehand!");
}

} // namespace python
} // namespace proxddp
