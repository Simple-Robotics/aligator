#include "proxddp/python/fwd.hpp"

#include "proxddp/stage-model.hpp"


namespace proxddp
{
namespace python
{
  void exposeNode()
  {
    using context::Scalar;
    using context::Manifold;
    using context::DynamicsModel;
    using StageType = StageModelTpl<Scalar>;

    bp::class_<StageType>(
      "StageModel", "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::init<const Manifold&,
               const int,
               const Manifold&,
               const DynamicsModel&
               >(bp::args("self", "space1", "nu", "space2", "dyn_model"))
    )
      .def(bp::init<const Manifold&,
                    const int,
                    const DynamicsModel&
                    >(bp::args("self", "space", "nu", "dyn_model")))
      .def("add_constraint", (void(StageType::*)(const StageType::ConstraintPtr&))&StageType::addConstraint)
      .def("create_data", &StageType::createData, "Create the data object.")
      // .def_readonly("space1", &StageModelTpl<Scalar>::xspace1_)
      .def_readonly("uspace", &StageType::uspace)
      .add_property("nu", &StageType::nu, "Control space dimension.")
    ;


    using StageData = StageType::Data;
    bp::class_<StageData>(
      "StageData", "Data struct for StageModel objects.",
      bp::init<const StageType&>()
    )
      .def_readonly("dyn_data", &StageData::dyn_data)
      .def_readonly("constraint_data", &StageData::constraint_data)
      ;

  }
  
} // namespace python
} // namespace proxddp

