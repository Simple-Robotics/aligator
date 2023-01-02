/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/compat/croco.hpp"
#include "proxddp/python/utils.hpp"

namespace proxddp {
namespace python {

void exposeCrocoddylCompat() {
  using context::Scalar;
  namespace ns_croc = ::proxddp::compat::croc;
  bp::def("convertCrocoddylProblem",
          &ns_croc::convertCrocoddylProblem<context::Scalar>,
          bp::args("croc_problem"),
          "Convert a Crocoddyl problem to a ProxDDP problem.");

  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using ActionModelWrapper = ns_croc::CrocActionModelWrapperTpl<Scalar>;
  using ActionDataWrapper = ns_croc::CrocActionDataWrapperTpl<Scalar>;

  bp::register_ptr_to_python<shared_ptr<ActionModelWrapper>>();
  bp::class_<ActionModelWrapper, bp::bases<context::StageModel>>(
      "ActionModelWrapper", "Wrapper for Crocoddyl action models.",
      bp::init<boost::shared_ptr<CrocActionModel>>(bp::args("action_model")))
      .def_readonly("action_model", &ActionModelWrapper::action_model_,
                    "Underlying Crocoddyl ActionModel.");

  bp::register_ptr_to_python<shared_ptr<ActionDataWrapper>>();
  bp::class_<ActionDataWrapper, bp::bases<context::StageData>>(
      "ActionDataWrapper", bp::no_init)
      .def_readonly("croc_data", &ActionDataWrapper::croc_data,
                    "Underlying Crocoddyl action data.");
}

} // namespace python
} // namespace proxddp
