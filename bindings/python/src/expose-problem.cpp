/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/core/traj-opt-problem.hpp"

namespace aligator {
namespace python {
void exposeProblem() {
  using context::ConstVectorRef;
  using context::CostBase;
  using context::Manifold;
  using context::Scalar;
  using context::StageData;
  using context::StageModel;
  using context::TrajOptData;
  using context::TrajOptProblem;
  using context::UnaryFunction;

  bp::class_<TrajOptProblem>("TrajOptProblem", "Define a shooting problem.",
                             bp::no_init)
      .def(bp::init<shared_ptr<UnaryFunction>,
                    const std::vector<shared_ptr<StageModel>> &,
                    shared_ptr<CostBase>>(
          "Constructor adding the initial constraint explicitly.",
          ("self"_a, "init_constraint", "stages", "term_cost")))
      .def(bp::init<ConstVectorRef, const std::vector<shared_ptr<StageModel>> &,
                    shared_ptr<CostBase>>(
          "Constructor for an initial value problem.",
          ("self"_a, "x0", "stages", "term_cost")))
      .def(bp::init<shared_ptr<UnaryFunction>, shared_ptr<CostBase>>(
          "Constructor adding the initial constraint explicitly (without "
          "stages).",
          ("self"_a, "init_constraint", "term_cost")))
      .def(bp::init<ConstVectorRef, const int, shared_ptr<Manifold>,
                    shared_ptr<CostBase>>(
          "Constructor for an initial value problem (without pre-allocated "
          "stages).",
          ("self"_a, "x0", "nu", "space", "term_cost")))
      .def<void (TrajOptProblem::*)(const shared_ptr<StageModel> &)>(
          "addStage", &TrajOptProblem::addStage, ("self"_a, "new_stage"),
          "Add a stage to the problem.")
      .def_readonly("stages", &TrajOptProblem::stages_,
                    "Stages of the shooting problem.")
      .def_readwrite("term_cost", &TrajOptProblem::term_cost_,
                     "Problem terminal cost.")
      .def_readwrite("term_constraints", &TrajOptProblem::term_cstrs_,
                     "Set of terminal constraints.")
      .add_property("num_steps", &TrajOptProblem::numSteps,
                    "Number of stages in the problem.")
      .add_property("x0_init", &TrajOptProblem::getInitState,
                    &TrajOptProblem::setInitState, "Initial state.")
      .add_property("init_constraint", &TrajOptProblem::init_condition_,
                    "Get initial state constraint.")
      .def("addTerminalConstraint", &TrajOptProblem::addTerminalConstraint,
           ("self"_a, "constraint"), "Add a terminal constraint.")
      .def("removeTerminalConstraint",
           &TrajOptProblem::removeTerminalConstraints, "self"_a,
           "Remove all terminal constraints.")
      .def("evaluate", &TrajOptProblem::evaluate,
           ("self"_a, "xs", "us", "prob_data", "num_threads"_a = 1),
           "Evaluate the problem costs, dynamics, and constraints.")
      .def("computeDerivatives", &TrajOptProblem::computeDerivatives,
           ("self"_a, "xs", "us", "prob_data", "num_threads"_a = 1),
           "Evaluate the problem derivatives. Call `evaluate()` first.")
      .def("replaceStageCircular", &TrajOptProblem::replaceStageCircular,
           ("self"_a, "model"),
           "Circularly replace the last stage in the problem, dropping the "
           "first stage.");

  bp::register_ptr_to_python<shared_ptr<TrajOptData>>();
  bp::class_<TrajOptData>(
      "TrajOptData", "Data struct for shooting problems.",
      bp::init<const TrajOptProblem &>(("self"_a, "problem")))
      .def_readwrite("init_data", &TrajOptData::init_data,
                     "Initial stage contraint data.")
      .def_readwrite("cost", &TrajOptData::cost_,
                     "Current cost of the TO problem.")
      .def_readwrite("term_cost", &TrajOptData::term_cost_data,
                     "Terminal cost data.")
      .def_readwrite("term_constraint", &TrajOptData::term_cstr_data,
                     "Terminal constraint data.")
      .def_readonly("stage_data", &TrajOptData::stage_data,
                    "Data for each stage.");
}

} // namespace python
} // namespace aligator
