#include "proxddp/python/fwd.hpp"
#include "proxddp/core/problem.hpp"


namespace proxddp
{
  namespace python
  {
    void exposeProblem()
    {
      using context::ShootingProblem;
      using StageModel = StageModelTpl<context::Scalar>;
      bp::class_<ShootingProblem>(
        "ShootingProblem", "Define a shooting problem.",
        bp::init<>()
      )
        .def("add_stage",
             (void(ShootingProblem::*)(const shared_ptr<StageModel>&))&ShootingProblem::addStage
             )
        .add_property("num_stages", &ShootingProblem::numStages)
        .def("createData", &ShootingProblem::createData)
      ;

    }
    
  } // namespace python
} // namespace proxddp

