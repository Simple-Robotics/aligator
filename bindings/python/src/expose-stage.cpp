/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/python/polymorphic-convertible.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/core/stage-model.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/core/cost-abstract.hpp"

#include <eigenpy/deprecation-policy.hpp>

namespace aligator {
namespace python {

void exposeStageData() {
  using context::StageData;
  using context::StageModel;

  bp::register_ptr_to_python<shared_ptr<StageData>>();
  StdVectorPythonVisitor<std::vector<shared_ptr<StageData>>, true>::expose(
      "StdVec_StageData");

  bp::class_<StageData>("StageData", "Data struct for StageModel objects.",
                        bp::init<const StageModel &>())
      .def_readonly("cost_data", &StageData::cost_data)
      .def_readwrite("dynamics_data", &StageData::dynamics_data)
      .def_readwrite("constraint_data", &StageData::constraint_data);
}

void exposeStage() {
  using context::ConstraintSet;
  using context::Manifold;
  using context::Scalar;
  using context::StageModel;

  using PolyCost = xyz::polymorphic<context::CostAbstract>;
  using PolyDynamics = xyz::polymorphic<context::DynamicsModel>;
  using PolyFunction = xyz::polymorphic<context::StageFunction>;
  using PolyCstrSet = xyz::polymorphic<ConstraintSet>;
  using PolyStage = xyz::polymorphic<StageModel>;

  register_polymorphic_to_python<PolyStage>();

  using StageVec = std::vector<PolyStage>;
  StdVectorPythonVisitor<StageVec, true>::expose(
      "StdVec_StageModel",
      eigenpy::details::overload_base_get_item_for_std_vector<StageVec>());

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  bp::class_<StageModel>(
      "StageModel",
      "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::no_init)
      .def(bp::init<const PolyCost &, const PolyDynamics &>(
          ("self"_a, "cost", "dynamics")))
      .def<void (StageModel::*)(const context::StageConstraint &)>(
          "addConstraint", &StageModel::addConstraint,
          eigenpy::deprecated_member<>("This method has been deprecated since "
                                       "StageConstraint is deprecated."),
          ("self"_a, "constraint"), "Add an existing constraint to the stage.")
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
      .def(PrintableVisitor<StageModel>())
      .def(CopyableVisitor<StageModel>())
      .def(PolymorphicVisitor<PolyStage>());
#pragma GCC diagnostic pop

  exposeStageData();
}

} // namespace python
} // namespace aligator
