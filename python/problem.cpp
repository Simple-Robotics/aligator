#include "proxddp/python/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"

namespace proxddp {
namespace python {
void exposeProblem() {
  using context::CostBase;
  using context::StageModel;
  using context::TrajOptData;
  using context::TrajOptProblem;

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
      .def_readonly("stages", &TrajOptProblem::stages_,
                    "Stages of the shooting problem.")
      .def_readwrite("term_cost", &TrajOptProblem::term_cost_,
                     "Problem terminal cost.")
      .add_property("num_steps", &TrajOptProblem::numSteps,
                    "Number of stages in the problem.")
      .add_property("x0", &TrajOptProblem::x0_init_, "Initial state.")
      .def("setTerminalConstraint", &TrajOptProblem::setTerminalConstraint,
           "Set terminal constraint.")
      .def("evaluate", &TrajOptProblem::evaluate,
           bp::args("self", "xs", "us", "prob_data"),
           "Rollout the problem costs, dynamics, and constraints.")
      .def("computeDerivatives", &TrajOptProblem::computeDerivatives,
           bp::args("self", "xs", "us", "prob_data"),
           "Rollout the problem derivatives.");

  bp::register_ptr_to_python<shared_ptr<TrajOptData>>();
  bp::class_<TrajOptData>(
      "TrajOptData", "Data struct for shooting problems.",
      bp::init<const TrajOptProblem &>(bp::args("self", "problem")))
      .def_readonly("term_cost", &TrajOptData::term_cost_data,
                    "Terminal cost data.")
      .def_readonly("term_constraint", &TrajOptData::term_cstr_data,
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
