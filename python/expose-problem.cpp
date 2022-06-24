#include "proxddp/python/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"


namespace proxddp
{
  namespace python
  {
    void exposeProblem()
    {
      using context::TrajOptProblem;
      using context::StageModel;
      using context::ProblemData;
      using context::CostBase;

      bp::class_<TrajOptProblem>(
        "TrajOptProblem", "Define a shooting problem.",
        bp::init<const context::VectorXs&, const std::vector<StageModel>&, const shared_ptr<CostBase>&>(
          bp::args("self", "x0", "stages", "term_cost")
          )
      )
        .def(bp::init<const context::VectorXs&, const int, const context::Manifold&, const shared_ptr<CostBase>&>(
              bp::args("self", "x0", "nu", "space", "term_cost"))
             )
        .def<void(TrajOptProblem::*)(const StageModel&)>(
          "addStage",
          &TrajOptProblem::addStage,
          bp::args("self", "new_stage"),
          "Add a stage to the problem.")
        .def_readwrite("stages", &TrajOptProblem::stages_, "Stages of the shooting problem.")
        .add_property("num_steps", &TrajOptProblem::numSteps, "Number of stages in the problem.")
        .add_property("x0", &TrajOptProblem::x0_init_, "Initial state.")
        .def("evaluate", &TrajOptProblem::evaluate,
             bp::args("self", "xs", "us", "prob_data"),
             "Rollout the problem costs, dynamics, and constraints.")
        .def("computeDerivatives", &TrajOptProblem::computeDerivatives,
             bp::args("self", "xs", "us", "prob_data"),
             "Rollout the problem derivatives.")
        .def(CreateDataPythonVisitor<TrajOptProblem>());

      bp::register_ptr_to_python<shared_ptr<ProblemData>>();
      bp::class_<ProblemData>(
        "TrajOptData", "Data struct for shooting problems.",
        bp::init<const TrajOptProblem&>(bp::args("self", "problem"))
      )
        .add_property("stage_data", bp::make_getter(&ProblemData::stage_data, bp::return_value_policy<bp::return_by_value>()),
                      "Data for each stage.")
      ;

      bp::def("computeTrajectoryCost", &computeTrajectoryCost<context::Scalar>,
              bp::args("problem", "data"),
              "Compute the cost of the trajectory. NOTE: problem.evaluate() must be called beforehand!");

    }
    
  } // namespace python
} // namespace proxddp

