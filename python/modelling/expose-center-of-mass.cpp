/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#ifdef PROXDDP_WITH_PINOCCHIO
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/functions.hpp"
#include "proxddp/python/modelling/multibody-utils.hpp"

#include "proxddp/python/modelling/multibody-utils.hpp"
#include "proxddp/modelling/multibody/center-of-mass-translation.hpp"
#include "proxddp/modelling/multibody/center-of-mass-velocity.hpp"

namespace proxddp {
namespace python {

using context::PinData;
using context::PinModel;
using context::Scalar;
using context::StageFunctionData;
using context::UnaryFunction;

void exposeCenterOfMassFunctions() {
  using CenterOfMassTranslation = CenterOfMassTranslationResidualTpl<Scalar>;
  using CenterOfMassTranslationData = CenterOfMassTranslationDataTpl<Scalar>;

  using CenterOfMassVelocity = CenterOfMassVelocityResidualTpl<Scalar>;
  using CenterOfMassVelocityData = CenterOfMassVelocityDataTpl<Scalar>;

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
}

} // namespace python
} // namespace proxddp

#endif
