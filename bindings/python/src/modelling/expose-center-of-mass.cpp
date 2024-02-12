/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/python/fwd.hpp"
#include "aligator/python/functions.hpp"
#include "aligator/python/modelling/multibody-utils.hpp"

#include "aligator/python/modelling/multibody-utils.hpp"
#include "aligator/modelling/multibody/center-of-mass-translation.hpp"
#include "aligator/modelling/multibody/center-of-mass-velocity.hpp"
#include "aligator/modelling/multibody/angular-momentum-constraint.hpp"
#include "aligator/modelling/contact-map.hpp"

namespace aligator {
namespace python {

using context::PinData;
using context::PinModel;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;
using ContactMap = ContactMapTpl<Scalar>;

void exposeCenterOfMassFunctions() {
  using CenterOfMassTranslation = CenterOfMassTranslationResidualTpl<Scalar>;
  using CenterOfMassTranslationData = CenterOfMassTranslationDataTpl<Scalar>;

  using CenterOfMassVelocity = CenterOfMassVelocityResidualTpl<Scalar>;
  using CenterOfMassVelocityData = CenterOfMassVelocityDataTpl<Scalar>;

  using AngularMomentumConstraintResidual =
      AngularMomentumConstraintResidualTpl<Scalar>;
  using AngularMomentumConstraintData =
      AngularMomentumConstraintDataTpl<Scalar>;

  bp::class_<CenterOfMassTranslation, bp::bases<UnaryFunction>>(
      "CenterOfMassTranslationResidual",
      "A residual function :math:`r(x) = com(x)` ",
      bp::init<const int, const int, shared_ptr<PinModel>,
               const context::Vector3s>(
          bp::args("self", "ndx", "nu", "model", "p_ref")))
      .def(FrameAPIVisitor<CenterOfMassTranslation>())
      .def("getReference", &CenterOfMassTranslation::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target Center Of Mass translation.")
      .def("setReference", &CenterOfMassTranslation::setReference,
           bp::args("self", "p_new"),
           "Set the target Center Of Mass translation.");

  bp::register_ptr_to_python<shared_ptr<CenterOfMassTranslationData>>();

  bp::class_<CenterOfMassTranslationData, bp::bases<StageFunctionData>>(
      "CenterOfMassTranslationResidualData",
      "Data Structure for CenterOfMassTranslation", bp::no_init)
      .def_readonly("pin_data", &CenterOfMassTranslationData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<CenterOfMassVelocity, bp::bases<UnaryFunction>>(
      "CenterOfMassVelocityResidual",
      "A residual function :math:`r(x) = vcom(x)` ",
      bp::init<const int, const int, shared_ptr<PinModel>,
               const context::Vector3s>(
          bp::args("self", "ndx", "nu", "model", "v_ref")))
      .def(FrameAPIVisitor<CenterOfMassVelocity>())
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

  bp::class_<AngularMomentumConstraintResidual, bp::bases<StageFunction>>(
      "AngularMomentumConstraintResidual",
      "A residual function :math:`r(x) = L - A(q) v` ",
      bp::init<const shared_ptr<PinModel> &, const context::Vector3s &,
               const ContactMap &>(
          bp::args("self", "model", "gravity", "contact_map")))
      .def(FrameAPIVisitor<AngularMomentumConstraintResidual>());

  bp::register_ptr_to_python<shared_ptr<AngularMomentumConstraintData>>();

  bp::class_<AngularMomentumConstraintData, bp::bases<StageFunctionData>>(
      "AngularMomentumConstraintResidualData",
      "Data Structure for AngularMomentumConstraint", bp::no_init)
      .def_readonly("pin_data", &AngularMomentumConstraintData::pin_data_,
                    "Pinocchio data struct.")
      .def_readonly("centroidal_data",
                    &AngularMomentumConstraintData::centroidal_data_,
                    "Centroidal data struct.");
}

} // namespace python
} // namespace aligator

#endif
