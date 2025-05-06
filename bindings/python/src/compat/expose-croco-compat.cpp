/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_CROCODDYL_COMPAT
#include "aligator/python/fwd.hpp"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/compat/crocoddyl/action-model-wrap.hpp"
#include "aligator/compat/crocoddyl/problem-wrap.hpp"
#include "aligator/compat/crocoddyl/state-wrap.hpp"
#endif

namespace aligator {
namespace python {

void exposeCrocoddylCompat() {
  bp::import("crocoddyl");

  using context::Scalar;
  namespace ns_croc = ::aligator::compat::croc;
  bp::def("convertCrocoddylProblem",
          &ns_croc::convertCrocoddylProblem<context::Scalar>,
          bp::args("croc_problem"),
          "Convert a Crocoddyl problem to an aligator problem.");

  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using StateAbstract = crocoddyl::StateAbstractTpl<Scalar>;
  using ns_croc::context::ActionDataWrapper;
  using ns_croc::context::ActionModelWrapper;
  using ns_croc::context::DynamicsDataWrapper;
  using ns_croc::context::StateWrapper;

  bp::class_<ActionModelWrapper, bp::bases<context::StageModel>>(
      "ActionModelWrapper", "Wrapper for Crocoddyl action models.",
      bp::init<shared_ptr<CrocActionModel>>(bp::args("action_model")))
      .def_readonly("action_model", &ActionModelWrapper::action_model_,
                    "Underlying Crocoddyl ActionModel.")
      .def(PolymorphicMultiBaseVisitor<context::StageModel>());

  bp::register_ptr_to_python<shared_ptr<ActionDataWrapper>>();
  bp::class_<ActionDataWrapper, bp::bases<context::StageData>>(
      "ActionDataWrapper", bp::no_init)
      .def(bp::init<const ActionModelWrapper &>(
          bp::args("self", "croc_action_model")))
      .def_readonly("croc_action_data", &ActionDataWrapper::croc_action_data,
                    "Underlying Crocoddyl action data.");

  bp::class_<DynamicsDataWrapper, bp::bases<context::StageFunctionData>>(
      "DynamicsDataWrapper", bp::no_init)
      .def(bp::init<const CrocActionModel &>(bp::args("self", "action_model")));

  bp::class_<StateWrapper, bp::bases<context::Manifold>>(
      "StateWrapper", "Wrapper for a Crocoddyl state.", bp::no_init)
      .def(bp::init<shared_ptr<StateAbstract>>(bp::args("self", "state")))
      .def_readonly("croc_state", &StateWrapper::croc_state)
      .def(PolymorphicMultiBaseVisitor<context::Manifold>());
}

} // namespace python
} // namespace aligator
#endif
