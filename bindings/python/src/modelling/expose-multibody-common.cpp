/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/multibody-common.hpp"

namespace aligator {
namespace python {

void exposeMultibodyCommon() {
  using namespace aligator::dynamics;
  using context::Scalar;

  using CommonModel = CommonModelTpl<Scalar>;
  using CommonModelData = CommonModelDataTpl<Scalar>;
  using CommonModelBuilder = CommonModelBuilderTpl<Scalar>;
  using MultibodyCommon = MultibodyCommonTpl<Scalar>;
  using MultibodyCommonData = MultibodyCommonDataTpl<Scalar>;
  using MultibodyCommonBuilder = MultibodyCommonBuilderTpl<Scalar>;

  bp::register_ptr_to_python<shared_ptr<MultibodyCommon>>();
  bp::class_<MultibodyCommon, bp::bases<CommonModel>>(
      "MultibodyCommon", "Compute constaint forward dynamics using Pinocchio.",
      bp::no_init)
      .def_readwrite("pin_model", &MultibodyCommon::pin_model_)
      .def_readwrite("actuation_matrix", &MultibodyCommon::actuation_matrix_)
      .def_readwrite("run_aba", &MultibodyCommon::run_aba_);

  bp::register_ptr_to_python<shared_ptr<MultibodyCommonData>>();
  bp::class_<MultibodyCommonData, bp::bases<CommonModelData>>(
      "MultibodyCommonData", "Store MultibodyCommon data", bp::no_init)
      .def_readwrite("tau", &MultibodyCommonData::tau_)
      .def_readwrite("qdd", &MultibodyCommonData::qdd_)
      .def_readwrite("qdd_dq", &MultibodyCommonData::qdd_dq_)
      .def_readwrite("qdd_dv", &MultibodyCommonData::qdd_dv_)
      .def_readwrite("qdd_dtau", &MultibodyCommonData::qdd_dtau_)
      .def_readwrite("pin_data", &MultibodyCommonData::pin_data_);

  bp::register_ptr_to_python<shared_ptr<MultibodyCommonBuilder>>();
  bp::class_<MultibodyCommonBuilder, bp::bases<CommonModelBuilder>>(
      "MultibodyCommonBuilder", "MultibodyCommon builder")
      .def("withPinocchioModel", &MultibodyCommonBuilder::withPinocchioModel,
           bp::return_internal_reference<>(), bp::args("self", "model"))
      .def("withActuationMatrix", &MultibodyCommonBuilder::withActuationMatrix,
           bp::return_internal_reference<>(),
           bp::args("self", "actuation_matrix"))
      .def("withRunAba", &MultibodyCommonBuilder::withRunAba,
           bp::return_internal_reference<>(), bp::args("self", "activated"));
}

} // namespace python
} // namespace aligator
