/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include <proxsuite-nlp/python/polymorphic.hpp>
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

  using PolyCost = xyz::polymorphic<context::CostAbstract>;
  using PolyDynamics = xyz::polymorphic<context::DynamicsModel>;
  using PolyFunction = xyz::polymorphic<context::StageFunction>;
  using PolyCstrSet = xyz::polymorphic<ConstraintSet>;
  using PolyStage = xyz::polymorphic<StageModel>;

  proxsuite::nlp::python::register_polymorphic_to_python<PolyStage>();
  bp::implicitly_convertible<StageModel, PolyStage>();

  StdVectorPythonVisitor<std::vector<PolyStage>, true>::expose(
      "StdVec_StageModel");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  bp::class_<StageModel>(
      "StageModel",
      "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::no_init)
      .def(bp::init<const PolyCost &, const PolyDynamics &>(
          ("self"_a, "cost", "dynamics")))
      .def<void (StageModel::*)(const context::StageConstraint &)>(
          "addConstraint", &StageModel::addConstraint, ("self"_a, "constraint"),
          "Add an existing constraint to the stage.")
      .def<void (StageModel::*)(const PolyFunction &, const PolyCstrSet &)>(
          "addConstraint", &StageModel::addConstraint,
          ("self"_a, "func", "cstr_set"),
          "Constructs a new constraint (from the underlying function and set) "
          "and adds it to the stage.")
      .def_readonly("constraints", &StageModel::constraints_,
                    "Get the set of constraints.")
      .def_readonly("dynamics", &StageModel::dynamics_, "Stage dynamics.")
      .add_property("xspace",
                    bp::make_getter(&StageModel::xspace_,
                                    bp::return_internal_reference<>()),
                    "State space for the current state :math:`x_k`.")
      .add_property("xspace_next",
                    bp::make_getter(&StageModel::xspace_next_,
                                    bp::return_internal_reference<>()),
                    "State space corresponding to next state :math:`x_{k+1}`.")
      .add_property("uspace",
                    bp::make_getter(&StageModel::uspace_,
                                    bp::return_internal_reference<>()),
                    "Control space.")
      .add_property("cost",
                    bp::make_getter(&StageModel::cost_,
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
      .def("evaluate", &StageModel::evaluate, ("self"_a, "x", "u", "y", "data"),
           "Evaluate the stage cost, dynamics, constraints.")
      .def("computeFirstOrderDerivatives",
           &StageModel::computeFirstOrderDerivatives,
           ("self"_a, "x", "u", "y", "data"),
           "Compute gradients of the stage cost and jacobians of the dynamics "
           "and "
           "constraints.")
      .def("computeSecondOrderDerivatives",
           &StageModel::computeSecondOrderDerivatives,
           ("self"_a, "x", "u", "data"), "Compute Hessians of the stage cost.")
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
#pragma GCC diagnostic pop

  bp::register_ptr_to_python<shared_ptr<StageData>>();
  StdVectorPythonVisitor<std::vector<shared_ptr<StageData>>, true>::expose(
      "StdVec_StageData");

  bp::class_<StageData>("StageData", "Data struct for StageModel objects.",
                        bp::init<const StageModel &>())
      .def_readonly("cost_data", &StageData::cost_data)
      .def_readwrite("dynamics_data", &StageData::dynamics_data)
      .def_readwrite("constraint_data", &StageData::constraint_data)
      .def(ClonePythonVisitor<StageData>());
}

} // namespace python
} // namespace aligator
