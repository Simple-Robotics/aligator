#include "proxddp/python/fwd.hpp"

#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
namespace python
{
  void exposeNode()
  {
    using context::Scalar;
    using context::Manifold;
    using context::DynamicsModel;
    using StageModel = StageModelTpl<Scalar>;

    bp::class_<StageModel>(
      "StageModel", "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::init<const Manifold&,
               const int,
               const Manifold&,
               const context::CostBase&,
               const DynamicsModel&
               >(bp::args("self", "space1", "nu", "space2", "cost", "dyn_model"))
    )
      .def(bp::init<const Manifold&,
                    const int,
                    const context::CostBase&,
                    const DynamicsModel&
                    >(bp::args("self", "space", "nu", "cost", "dyn_model")))
      .def("add_constraint", (void(StageModel::*)(const StageModel::ConstraintPtr&))&StageModel::addConstraint)
      .def("createData", &StageModel::createData, "Create the data object.")
      .def_readonly("uspace", &StageModel::uspace)
      .def("evaluate", &StageModel::evaluate,
           bp::args("self", "x", "u", "y", "data"),
           "Evaluate the stage cost, dynamics, constraints.")
      .def("computeDerivatives", &StageModel::computeDerivatives,
           bp::args("self", "x", "u", "y", "lbdas", "data", "compute_all_hessians"),
           "Compute derivatives of the stage cost, dynamics, and constraints.")
      .add_property("ndx1", &StageModel::ndx1)
      .add_property("ndx2", &StageModel::ndx2)
      .add_property("nu", &StageModel::nu, "Control space dimension.")
    ;

    using StageData = StageModel::Data;
    bp::register_ptr_to_python<shared_ptr<StageData>>();
    bp::class_<StageData>(
      "StageData", "Data struct for StageModel objects.",
      bp::init<const StageModel&>()
    )
      .def_readonly("cost_data", &StageData::cost_data)
      .def_readonly("dyn_data", &StageData::dyn_data)
      .def_readonly("constraint_data", &StageData::constraint_data)
      ;

  }
  
} // namespace python
} // namespace proxddp

