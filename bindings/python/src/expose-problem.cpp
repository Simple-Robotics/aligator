/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/traj-opt-data.hpp"
#include "aligator/core/cost-abstract.hpp"
#include <eigenpy/deprecation-policy.hpp>

namespace aligator {
namespace python {
void exposeProblem() {
  using context::ConstVectorRef;
  using context::CostAbstract;
  using context::Manifold;
  using context::Scalar;
  using context::StageData;
  using context::StageModel;
  using context::TrajOptData;
  using context::TrajOptProblem;
  using context::UnaryFunction;

  using PolyUnaryFunction = xyz::polymorphic<UnaryFunction>;
  using PolyFunction = xyz::polymorphic<context::StageFunction>;
  using PolyStage = xyz::polymorphic<StageModel>;
  using PolyCost = xyz::polymorphic<CostAbstract>;
  using PolyManifold = xyz::polymorphic<Manifold>;
  using PolySet = xyz::polymorphic<context::ConstraintSet>;

  bp::class_<TrajOptProblem>("TrajOptProblem", "Define a shooting problem.",
                             bp::no_init)
      .def(
          bp::init<PolyUnaryFunction, const std::vector<PolyStage> &, PolyCost>(
              "Constructor adding the initial constraint explicitly.",
              ("self"_a, "init_constraint", "stages", "term_cost")))
      .def(bp::init<ConstVectorRef, const std::vector<PolyStage> &, PolyCost>(
          "Constructor for an initial value problem.",
          ("self"_a, "x0", "stages", "term_cost")))
      .def(bp::init<PolyUnaryFunction, PolyCost>(
          "Constructor adding the initial constraint explicitly (without "
          "stages).",
          ("self"_a, "init_constraint", "term_cost")))
      .def(bp::init<ConstVectorRef, const int, PolyManifold, PolyCost>(
          "Constructor for an initial value problem (without pre-allocated "
          "stages).",
          ("self"_a, "x0", "nu", "space", "term_cost")))
      .def<void (TrajOptProblem::*)(const PolyStage &)>(
          "addStage", &TrajOptProblem::addStage, ("self"_a, "new_stage"),
          "Add a stage to the problem.")
      .def_readonly("stages", &TrajOptProblem::stages_,
                    "Stages of the shooting problem.")
      .def_readwrite("term_cost", &TrajOptProblem::term_cost_,
                     "Problem terminal cost.")
      .def_readwrite("term_constraints", &TrajOptProblem::term_cstrs_,
                     "Set of terminal constraints.")
      .add_property("num_steps", &TrajOptProblem::numSteps,
                    "Number of stages in the problem.CostPtr")
      .add_property("x0_init", &TrajOptProblem::getInitState,
                    &TrajOptProblem::setInitState, "Initial state.")
      .add_property("init_constraint", &TrajOptProblem::init_constraint_,
                    "Get initial state constraint.")
      .def<void (TrajOptProblem::*)(const PolyFunction &, const PolySet &)>(
          "addTerminalConstraint", &TrajOptProblem::addTerminalConstraint,
          ("self"_a, "func", "set"), "Add a terminal constraint.")
      .def("removeTerminalConstraint",
           &TrajOptProblem::removeTerminalConstraints,
           eigenpy::deprecated_member<>(
               "This method is deprecated (due to a typo which was fixed). Use "
               "removeTerminalConstraints instead."),
           ("self"_a), "Remove all terminal constraints.")
      .def("removeTerminalConstraints",
           &TrajOptProblem::removeTerminalConstraints, "self"_a,
           "Remove all terminal constraints.")
      .def("evaluate", &TrajOptProblem::evaluate,
           ("self"_a, "xs", "us", "prob_data", "num_threads"_a = 1),
           "Evaluate the problem costs, dynamics, and constraints.")
      .def("computeDerivatives", &TrajOptProblem::computeDerivatives,
           ("self"_a, "xs", "us", "prob_data", "num_threads"_a = 1,
            "compute_second_order"_a = true),
           "Evaluate the problem derivatives. Call `evaluate()` first.")
      .def("replaceStageCircular", &TrajOptProblem::replaceStageCircular,
           ("self"_a, "model"),
           "Circularly replace the last stage in the problem, dropping the "
           "first stage.")
      .def("checkIntegrity", &TrajOptProblem::checkIntegrity, ("self"_a));

  bp::def("computeTrajectoryCost", computeTrajectoryCost<Scalar>,
          ("problem_data"_a),
          "Compute trajectory cost (call evaluate() first!)");

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
