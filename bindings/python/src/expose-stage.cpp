/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/core/stage-model.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/core/cost-abstract.hpp"

#include <proxsuite-nlp/python/deprecation-policy.hpp>

namespace aligator {
namespace python {

void exposeStage() {
  using context::ConstraintSet;
  using context::Manifold;
  using context::Scalar;
  using context::StageModel;
  using StageData = StageDataTpl<Scalar>;

  using CostPtr = shared_ptr<context::CostAbstract>;
  using DynamicsPtr = shared_ptr<context::DynamicsModel>;
  using FunctionPtr = shared_ptr<context::StageFunction>;
  using CstrSetPtr = shared_ptr<ConstraintSet>;

  StdVectorPythonVisitor<std::vector<shared_ptr<StageModel>>, true>::expose(
      "StdVec_StageModel");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  bp::register_ptr_to_python<shared_ptr<StageModel>>();
  bp::class_<StageModel>(
      "StageModel",
      "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::init<CostPtr, DynamicsPtr>(bp::args("self", "cost", "dynamics")))
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
      .def_readonly("dynamics", &StageModel::dynamics_, "Stage dynamics.")
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
      .add_property(
          "dyn_model",
          bp::make_function(&StageModel::dyn_model,
                            proxsuite::nlp::deprecation_warning_policy<
                                proxsuite::nlp::DeprecationType::DEPRECATION,
                                bp::return_internal_reference<>>(
                                "Deprecated. Use StageModel.dynamics instead")),
          "Stage dynamics.")
      .def("configure", &StageModel::configure, bp::args("self"),
           "Configure cost, constraints and dynamics")
      .def("evaluate", &StageModel::evaluate,
           bp::args("self", "x", "u", "y", "data"),
           "Evaluate the stage cost, dynamics, constraints.")
      .def("computeFirstOrderDerivatives",
           &StageModel::computeFirstOrderDerivatives,
           bp::args("self", "x", "u", "y", "data"),
           "Compute gradients of the stage cost and jacobians of the dynamics "
           "and "
           "constraints.")
      .def("computeSecondOrderDerivatives",
           &StageModel::computeSecondOrderDerivatives,
           bp::args("self", "x", "u", "data"),
           "Compute Hessians of the stage cost.")
      .add_property("ndx1", &StageModel::ndx1)
      .add_property("ndx2", &StageModel::ndx2)
      .add_property("nu", &StageModel::nu, "Control space dimension.")
      .add_property("num_primal", &StageModel::numPrimal,
                    "Number of primal variables.")
      .add_property("num_dual", &StageModel::numDual,
                    "Number of dual variables.")
      .def("createData", &StageModel::createData, bp::args("self"),
           "Create a data object.")
      .def(ClonePythonVisitor<StageModel>())
      .def(PrintableVisitor<StageModel>());
#pragma GCC diagnostic pop

  bp::register_ptr_to_python<shared_ptr<StageData>>();
  StdVectorPythonVisitor<std::vector<shared_ptr<StageData>>, true>::expose(
      "StdVec_StageData");

  bp::class_<StageData>("StageData", "Data struct for StageModel objects.",
                        bp::init<const StageModel &>())
      .def_readonly("cost_data", &StageData::cost_data)
      .def_readwrite("dynamics_data", &StageData::dynamics_data)
      .def_readonly("common_model_data_container",
                    &StageData::common_model_data_container)
      .def_readwrite("constraint_data", &StageData::constraint_data)
      .def(ClonePythonVisitor<StageData>());
}

} // namespace python
} // namespace aligator
