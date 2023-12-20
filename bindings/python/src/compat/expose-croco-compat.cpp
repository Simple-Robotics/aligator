/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#ifdef PROXDDP_WITH_CROCODDYL_COMPAT
#include "proxddp/python/compat/croco.hpp"
#include "proxddp/python/utils.hpp"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/compat/crocoddyl/instantiate.txx"
#endif

namespace proxddp {
namespace python {

void exposeCrocoddylCompat() {
  bp::import("crocoddyl");

  using context::Scalar;
  namespace ns_croc = ::proxddp::compat::croc;
  bp::def("convertCrocoddylProblem",
          &ns_croc::convertCrocoddylProblem<context::Scalar>,
          bp::args("croc_problem"),
          "Convert a Crocoddyl problem to a ProxDDP problem.");

  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using StateAbstract = crocoddyl::StateAbstractTpl<Scalar>;
  using ns_croc::context::ActionDataWrapper;
  using ns_croc::context::ActionModelWrapper;
  using ns_croc::context::DynamicsDataWrapper;
  using ns_croc::context::StateWrapper;

  bp::register_ptr_to_python<shared_ptr<ActionModelWrapper>>();
  bp::class_<ActionModelWrapper, bp::bases<context::StageModel>>(
      "ActionModelWrapper", "Wrapper for Crocoddyl action models.",
      bp::init<boost::shared_ptr<CrocActionModel>>(bp::args("action_model")))
      .def_readonly("action_model", &ActionModelWrapper::action_model_,
                    "Underlying Crocoddyl ActionModel.");

  bp::register_ptr_to_python<shared_ptr<ActionDataWrapper>>();
  bp::class_<ActionDataWrapper, bp::bases<context::StageData>>(
      "ActionDataWrapper", bp::no_init)
      .def(bp::init<const boost::shared_ptr<CrocActionModel> &>(
          bp::args("self", "croc_action_model")))
      .def_readonly("croc_action_data", &ActionDataWrapper::croc_action_data,
                    "Underlying Crocoddyl action data.");

  bp::class_<DynamicsDataWrapper, bp::bases<context::StageFunctionData>>(
      "DynamicsDataWrapper", bp::no_init)
      .def(bp::init<const CrocActionModel &>(bp::args("self", "action_model")));

  bp::class_<StateWrapper, bp::bases<context::Manifold>>(
      "StateWrapper", "Wrapper for a Crocoddyl state.", bp::no_init)
      .def(
          bp::init<boost::shared_ptr<StateAbstract>>(bp::args("self", "state")))
      .def_readonly("croc_state", &StateWrapper::croc_state);
}

} // namespace python
} // namespace proxddp
#endif
