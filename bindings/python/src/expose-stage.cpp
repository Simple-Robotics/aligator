/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"

#include "proxddp/core/stage-model.hpp"
#include "proxddp/core/stage-data.hpp"

namespace aligator {
namespace python {

void exposeStage() {
  using context::ConstraintSet;
  using context::Manifold;
  using context::Scalar;
  using context::StageModel;
  using StageData = StageDataTpl<Scalar>;

  using CostPtr = shared_ptr<context::CostBase>;
  using DynamicsPtr = shared_ptr<context::DynamicsModel>;
  using FunctionPtr = shared_ptr<context::StageFunction>;
  using CstrSetPtr = shared_ptr<ConstraintSet>;

  StdVectorPythonVisitor<std::vector<shared_ptr<StageModel>>, true>::expose(
      "StdVec_StageModel");

  bp::register_ptr_to_python<shared_ptr<StageModel>>();
  bp::class_<StageModel>(
      "StageModel",
      "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::init<CostPtr, DynamicsPtr>(bp::args("self", "cost", "dyn_model")))
      .def<void (StageModel::*)(const context::StageConstraint &)>(
          "addConstraint", &StageModel::addConstraint,
          bp::args("self", "constraint"),
          "Add an existing constraint to the stage.")
      .def<void (StageModel::*)(FunctionPtr, CstrSetPtr)>(
          "addConstraint", &StageModel::addConstraint,
          bp::args("self", "func", "cstr_set"),
          "Constructs a new constraint (from the underlying function and set) "
          "and adds it to the stage.")
      .def_readonly("constraints", &StageModel::constraints_,
                    "Get the set of constraints.")
      .add_property("xspace",
                    bp::make_function(&StageModel::xspace,
                                      bp::return_internal_reference<>()),
                    "State space for the current state :math:`x_k`.")
      .add_property("xspace_next",
                    bp::make_function(&StageModel::xspace_next,
                                      bp::return_internal_reference<>()),
                    "State space corresponding to next state :math:`x_{k+1}`.")
      .add_property("uspace",
                    bp::make_function(&StageModel::uspace,
                                      bp::return_internal_reference<>()),
                    "Control space.")
      .add_property("cost",
                    bp::make_function(&StageModel::cost,
                                      bp::return_internal_reference<>()),
                    "Stage cost.")
      .add_property("dyn_model",
                    bp::make_function(&StageModel::dyn_model,
                                      bp::return_internal_reference<>()),
                    "Stage dynamics.")
      .def("evaluate", &StageModel::evaluate,
           bp::args("self", "x", "u", "y", "data"),
           "Evaluate the stage cost, dynamics, constraints.")
      .def("computeDerivatives", &StageModel::computeDerivatives,
           bp::args("self", "x", "u", "y", "data"),
           "Compute derivatives of the stage cost, dynamics, and constraints.")
      .add_property("ndx1", &StageModel::ndx1)
      .add_property("ndx2", &StageModel::ndx2)
      .add_property("nu", &StageModel::nu, "Control space dimension.")
      .add_property("num_primal", &StageModel::numPrimal,
                    "Number of primal variables.")
      .add_property("num_dual", &StageModel::numDual,
                    "Number of dual variables.")
      .def(CreateDataPythonVisitor<StageModel>())
      .def(ClonePythonVisitor<StageModel>())
      .def(PrintableVisitor<StageModel>());

  bp::register_ptr_to_python<shared_ptr<StageData>>();
  StdVectorPythonVisitor<std::vector<shared_ptr<StageData>>, true>::expose(
      "StdVec_StageData");

  bp::class_<StageData>("StageData", "Data struct for StageModel objects.",
                        bp::init<const StageModel &>())
      .def_readonly("cost_data", &StageData::cost_data)
      .def_readwrite("constraint_data", &StageData::constraint_data)
      .def(ClonePythonVisitor<StageData>());
}

} // namespace python
} // namespace aligator
