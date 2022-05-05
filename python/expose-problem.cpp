#include "proxddp/python/fwd.hpp"
#include "proxddp/core/problem.hpp"


namespace proxddp
{
  namespace python
  {
    void exposeProblem()
    {
      using context::ShootingProblem;
      using context::StageModel;
      using context::ProblemData;

      bp::class_<ShootingProblem>(
        "ShootingProblem", "Define a shooting problem.",
        bp::init<>()
      )
        .def<void(ShootingProblem::*)(const StageModel&)>(
          "add_stage",
          &ShootingProblem::addStage,
          bp::args("self", "new_stage"),
          "Add a stage to the problem.")
        .def_readonly("stages", &ShootingProblem::stages_, "Stages of the shooting problem.")
        .add_property("num_steps", &ShootingProblem::numSteps, "Number of stages in the problem.")
        .def("evaluate", &ShootingProblem::evaluate,
             bp::args("self", "xs", "us", "prob_data"),
             "Rollout the problem costs, dynamics, and constraints.")
        .def("computeDerivatives", &ShootingProblem::computeDerivatives,
             bp::args("self", "xs", "us", "prob_data"),
             "Rollout the problem derivatives.")
        .def(CreateDataPythonVisitor<ShootingProblem>());

      bp::register_ptr_to_python<shared_ptr<ProblemData>>();
      bp::class_<ProblemData, boost::noncopyable>(
        "ProblemData", bp::no_init
      )
        .add_property("stage_data", bp::make_getter(&ProblemData::stage_data, bp::return_value_policy<bp::return_by_value>()),
                      "Data for each stage.")
      ;

    }
    
  } // namespace python
} // namespace proxddp

