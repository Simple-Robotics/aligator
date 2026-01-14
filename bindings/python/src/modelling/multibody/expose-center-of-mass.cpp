/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO

// Boost.Python 1.74 include manually mpl/vector/vector20.hpp
// that prevent us to define mpl::list and mpl::vector with
// the right size.
// To avoid this issue this header should be included first.
#include <pinocchio/fwd.hpp>

#include "aligator/python/fwd.hpp"
#include "aligator/python/modelling/multibody-utils.hpp"

#include "aligator/modelling/multibody/center-of-mass-translation.hpp"
#include "aligator/modelling/multibody/center-of-mass-velocity.hpp"
#include "aligator/modelling/multibody/dcm-position.hpp"
#include "aligator/modelling/multibody/centroidal-momentum.hpp"
#include "aligator/modelling/multibody/centroidal-momentum-derivative.hpp"

namespace aligator {
namespace python {

using context::PinData;
using context::PinModel;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;

void exposeCenterOfMassFunctions() {
  using CenterOfMassTranslation = CenterOfMassTranslationResidualTpl<Scalar>;
  using CenterOfMassTranslationData = CenterOfMassTranslationDataTpl<Scalar>;

  using CenterOfMassVelocity = CenterOfMassVelocityResidualTpl<Scalar>;
  using CenterOfMassVelocityData = CenterOfMassVelocityDataTpl<Scalar>;

  using DCMPosition = DCMPositionResidualTpl<Scalar>;
  using DCMPositionData = DCMPositionDataTpl<Scalar>;

  using CentroidalMomentumDerivativeResidual =
      CentroidalMomentumDerivativeResidualTpl<Scalar>;
  using CentroidalMomentumDerivativeData =
      CentroidalMomentumDerivativeDataTpl<Scalar>;

  using CentroidalMomentumResidual = CentroidalMomentumResidualTpl<Scalar>;
  using CentroidalMomentumData = CentroidalMomentumDataTpl<Scalar>;

  PolymorphicMultiBaseVisitor<StageFunction> func_visitor;
  PolymorphicMultiBaseVisitor<UnaryFunction, StageFunction> unary_visitor;

  bp::class_<CenterOfMassTranslation, bp::bases<UnaryFunction>>(
      "CenterOfMassTranslationResidual",
      "A residual function :math:`r(x) = com(x)` ",
      bp::init<const int, const int, const PinModel &, const context::Vector3s>(
          bp::args("self", "ndx", "nu", "model", "p_ref")))
      .def(FrameAPIVisitor<CenterOfMassTranslation>())
      .def("getReference", &CenterOfMassTranslation::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target Center Of Mass translation.")
      .def("setReference", &CenterOfMassTranslation::setReference,
           bp::args("self", "p_new"),
           "Set the target Center Of Mass translation.")
      .def(unary_visitor);

  bp::register_ptr_to_python<shared_ptr<CenterOfMassTranslationData>>();

  bp::class_<CenterOfMassTranslationData, bp::bases<StageFunctionData>>(
      "CenterOfMassTranslationResidualData",
      "Data Structure for CenterOfMassTranslation", bp::no_init)
      .def_readonly("pin_data", &CenterOfMassTranslationData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<CenterOfMassVelocity, bp::bases<UnaryFunction>>(
      "CenterOfMassVelocityResidual",
      "A residual function :math:`r(x) = vcom(x)` ",
      bp::init<const int, const int, const PinModel &, const context::Vector3s>(
          bp::args("self", "ndx", "nu", "model", "v_ref")))
      .def(FrameAPIVisitor<CenterOfMassVelocity>())
      .def(unary_visitor)
      .def("getReference", &CenterOfMassVelocity::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target Center Of Mass velocity.")
      .def("setReference", &CenterOfMassVelocity::setReference,
           bp::args("self", "p_new"),
           "Set the target Center Of Mass velocity.");

  bp::register_ptr_to_python<shared_ptr<CenterOfMassVelocityData>>();

  bp::class_<CenterOfMassVelocityData, bp::bases<StageFunctionData>>(
      "CenterOfMassVelocityResidualData",
      "Data Structure for CenterOfMassVelocity", bp::no_init)
      .def_readonly("pin_data", &CenterOfMassVelocityData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<DCMPosition, bp::bases<UnaryFunction>>(
      "DCMPositionResidual", "A residual function :math:`r(x) = dcm(x)` ",
      bp::init<const int, const int, const PinModel &,
               const context::Vector3s &, const double>(
          bp::args("self", "ndx", "nu", "model", "dcm_ref", "alpha")))
      .def(FrameAPIVisitor<DCMPosition>())
      .def(unary_visitor)
      .def("getReference", &DCMPosition::getReference, bp::args("self"),
           bp::return_internal_reference<>(), "Get the target DCM position.")
      .def("setReference", &DCMPosition::setReference,
           bp::args("self", "new_ref"), "Set the target DCM position.");

  bp::register_ptr_to_python<shared_ptr<DCMPositionData>>();

  bp::class_<DCMPositionData, bp::bases<StageFunctionData>>(
      "DCMPositionResidualData", "Data Structure for DCMPosition", bp::no_init)
      .def_readonly("pin_data", &DCMPositionData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<CentroidalMomentumDerivativeResidual, bp::bases<StageFunction>>(
      "CentroidalMomentumDerivativeResidual",
      "A residual function :math:`r(x) = H_dot` ",
      bp::init<const int, const PinModel &, const context::Vector3s &,
               const std::vector<bool> &,
               const std::vector<pinocchio::FrameIndex> &, const int>(
          bp::args("self", "ndx", "model", "gravity", "contact_states",
                   "contact_ids", "force_size")))
      .def(func_visitor)
      .def_readwrite("contact_states",
                     &CentroidalMomentumDerivativeResidual::contact_states_);

  bp::register_ptr_to_python<shared_ptr<CentroidalMomentumDerivativeData>>();

  bp::class_<CentroidalMomentumDerivativeData, bp::bases<StageFunctionData>>(
      "CentroidalMomentumDerivativeResidualData",
      "Data Structure for CentroidalMomentumDerivativeResidual", bp::no_init)
      .def_readonly("pin_data", &CentroidalMomentumDerivativeData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<CentroidalMomentumResidual, bp::bases<UnaryFunction>>(
      "CentroidalMomentumResidual", "A residual function :math:`r(x) = H` ",
      bp::init<const int, const int, const PinModel &,
               const context::Vector6s &>(
          bp::args("self", "ndx", "nu", "model", "h_ref")))
      .def(unary_visitor)
      .def("getReference", &CentroidalMomentumResidual::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the centroidal target.")
      .def("setReference", &CentroidalMomentumResidual::setReference,
           bp::args("self", "h_new"), "Set the centroidal target.");

  bp::register_ptr_to_python<shared_ptr<CentroidalMomentumData>>();

  bp::class_<CentroidalMomentumData, bp::bases<StageFunctionData>>(
      "CentroidalMomentumResidualData",
      "Data Structure for CentroidalMomentumResidual", bp::no_init)
      .def_readonly("pin_data", &CentroidalMomentumData::pin_data_,
                    "Pinocchio data struct.");
}

} // namespace python
} // namespace aligator

#endif
