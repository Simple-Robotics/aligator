/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#ifdef ALIGATOR_PINOCCHIO_V3

#include "aligator/modelling/dynamics/multibody-constraint-common.hpp"

namespace aligator {
namespace python {

void exposeMultibodyConstraintCommon() {
  using namespace aligator::dynamics;
  using context::Scalar;

  using CommonModel = CommonModelTpl<Scalar>;
  using CommonModelData = CommonModelDataTpl<Scalar>;
  using CommonModelBuilder = CommonModelBuilderTpl<Scalar>;
  using MultibodyConstraintCommon = MultibodyConstraintCommonTpl<Scalar>;
  using MultibodyConstraintCommonData =
      MultibodyConstraintCommonDataTpl<Scalar>;
  using MultibodyConstraintCommonBuilder =
      MultibodyConstraintCommonBuilderTpl<Scalar>;

  bp::register_ptr_to_python<shared_ptr<MultibodyConstraintCommon>>();
  bp::class_<MultibodyConstraintCommon, bp::bases<CommonModel>>(
      "MultibodyConstraintCommon",
      "Compute constaint forward dynamics using Pinocchio.", bp::no_init)
      .def_readwrite("pin_model", &MultibodyConstraintCommon::pin_model_)
      .def_readwrite("actuation_matrix",
                     &MultibodyConstraintCommon::actuation_matrix_)
      .def_readwrite("run_aba", &MultibodyConstraintCommon::run_aba_)
      .def_readwrite("constraint_models",
                     &MultibodyConstraintCommon::constraint_models_)
      .def_readwrite("prox_settings_",
                     &MultibodyConstraintCommon::prox_settings_);

  bp::register_ptr_to_python<shared_ptr<MultibodyConstraintCommonData>>();
  bp::class_<MultibodyConstraintCommonData, bp::bases<CommonModelData>>(
      "MultibodyConstraintCommonData", "Store MultibodyConstraintCommon data",
      bp::no_init)
      .def_readwrite("tau", &MultibodyConstraintCommonData::tau_)
      .def_readwrite("qdd", &MultibodyConstraintCommonData::qdd_)
      .def_readwrite("qdd_dtau", &MultibodyConstraintCommonData::qdd_dtau_)
      .def_readwrite("pin_data", &MultibodyConstraintCommonData::pin_data_)
      .def_readwrite("constraint_datas",
                     &MultibodyConstraintCommonData::constraint_datas_)
      .def_readwrite("prox_settings",
                     &MultibodyConstraintCommonData::prox_settings_);

  bp::register_ptr_to_python<shared_ptr<MultibodyConstraintCommonBuilder>>();
  bp::class_<MultibodyConstraintCommonBuilder, bp::bases<CommonModelBuilder>>(
      "MultibodyConstraintCommonBuilder", "MultibodyConstraintCommon builder")
      .def("withPinocchioModel",
           &MultibodyConstraintCommonBuilder::withPinocchioModel,
           bp::return_internal_reference<>(), bp::args("self", "model"))
      .def("withActuationMatrix",
           &MultibodyConstraintCommonBuilder::withActuationMatrix,
           bp::return_internal_reference<>(),
           bp::args("self", "actuation_matrix"))
      .def("withRunAba", &MultibodyConstraintCommonBuilder::withRunAba,
           bp::return_internal_reference<>(), bp::args("self", "activated"))
      .def("withConstraintModels",
           &MultibodyConstraintCommonBuilder::withConstraintModels,
           bp::return_internal_reference<>(),
           bp::args("self", "constraint_models"))
      .def("withProxSettings",
           &MultibodyConstraintCommonBuilder::withProxSettings,
           bp::return_internal_reference<>(), bp::args("self", "model"));
}

} // namespace python
} // namespace aligator

#endif // ALIGATOR_PINOCCHIO_V3
