/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#ifdef PROXDDP_WITH_PINOCCHIO
#include "proxddp/fwd.hpp"
#include "proxddp/python/functions.hpp"
#include "proxddp/python/modelling/multibody-utils.hpp"

#include "proxddp/modelling/multibody/frame-placement.hpp"
#include "proxddp/modelling/multibody/frame-velocity.hpp"
#include "proxddp/modelling/multibody/frame-translation.hpp"
#ifdef PROXDDP_PINOCCHIO_V3
#include "proxddp/modelling/multibody/constrained-rnea.hpp"
#endif

namespace proxddp {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::PinData;
using context::PinModel;

// fwd declaration, see expose-fly-high.cpp
void exposeFlyHigh();
void exposeCenterOfMassFunctions();

void exposeFrameFunctions() {
  using context::Manifold;
  using context::Scalar;
  using context::UnaryFunction;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Motion = pinocchio::MotionTpl<Scalar>;

  using FramePlacement = FramePlacementResidualTpl<Scalar>;
  using FramePlacementData = FramePlacementDataTpl<Scalar>;

  using FrameVelocity = FrameVelocityResidualTpl<Scalar>;
  using FrameVelocityData = FrameVelocityDataTpl<Scalar>;

  using FrameTranslation = FrameTranslationResidualTpl<Scalar>;
  using FrameTranslationData = FrameTranslationDataTpl<Scalar>;

  bp::register_ptr_to_python<shared_ptr<PinData>>();

  bp::class_<FramePlacement, bp::bases<UnaryFunction>>(
      "FramePlacementResidual", "Frame placement residual function.",
      bp::init<int, int, shared_ptr<PinModel>, const SE3 &,
               pinocchio::FrameIndex>(
          bp::args("self", "ndx", "nu", "model", "p_ref", "id")))
      .def(FrameAPIVisitor<FramePlacement>())
      .def("getReference", &FramePlacement::getReference, bp::args("self"),
           bp::return_internal_reference<>(), "Get the target frame in SE3.")
      .def("setReference", &FramePlacement::setReference,
           bp::args("self", "p_new"), "Set the target frame in SE3.");

  bp::register_ptr_to_python<shared_ptr<FramePlacementData>>();

  bp::class_<FramePlacementData, bp::bases<context::FunctionData>>(
      "FramePlacementData", "Data struct for FramePlacementResidual.",
      bp::no_init)
      .def_readonly("rMf", &FramePlacementData::rMf_, "Frame placement error.")
      .def_readonly("rJf", &FramePlacementData::rJf_)
      .def_readonly("fJf", &FramePlacementData::fJf_)
      .def_readonly("pin_data", &FramePlacementData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<FrameVelocity, bp::bases<UnaryFunction>>(
      "FrameVelocityResidual", "Frame velocity residual function.",
      bp::init<int, int, shared_ptr<PinModel>, const Motion &,
               pinocchio::FrameIndex, pinocchio::ReferenceFrame>(bp::args(
          "self", "ndx", "nu", "model", "v_ref", "id", "reference_frame")))
      .def(FrameAPIVisitor<FrameVelocity>())
      .def("getReference", &FrameVelocity::getReference, bp::args("self"),
           bp::return_internal_reference<>(), "Get the target frame velocity.")
      .def("setReference", &FrameVelocity::setReference,
           bp::args("self", "v_new"), "Set the target frame velocity.");

  bp::register_ptr_to_python<shared_ptr<FrameVelocityData>>();

  bp::class_<FrameVelocityData, bp::bases<context::FunctionData>>(
      "FrameVelocityData", "Data struct for FrameVelocityResidual.",
      bp::no_init)
      .def_readonly("pin_data", &FrameVelocityData::pin_data_,
                    "Pinocchio data struct.");

  bp::class_<FrameTranslation, bp::bases<UnaryFunction>>(
      "FrameTranslationResidual", "Frame placement residual function.",
      bp::init<int, int, shared_ptr<PinModel>, const context::Vector3s &,
               pinocchio::FrameIndex>(
          bp::args("self", "ndx", "nu", "model", "p_ref", "id")))
      .def(FrameAPIVisitor<FrameTranslation>())
      .def("getReference", &FrameTranslation::getReference, bp::args("self"),
           bp::return_internal_reference<>(),
           "Get the target frame translation.")
      .def("setReference", &FrameTranslation::setReference,
           bp::args("self", "p_new"), "Set the target frame translation.");

  bp::register_ptr_to_python<shared_ptr<FrameTranslationData>>();

  bp::class_<FrameTranslationData, bp::bases<context::FunctionData>>(
      "FrameTranslationData", "Data struct for FrameTranslationResidual.",
      bp::no_init)
      .def_readonly("fJf", &FrameTranslationData::fJf_)
      .def_readonly("pin_data", &FrameTranslationData::pin_data_,
                    "Pinocchio data struct.");
}

#ifdef PROXDDP_PINOCCHIO_V3
auto underactuatedConstraintInvDyn_proxy(const PinModel &model, PinData &data,
                                         const ConstVectorRef &q,
                                         const ConstVectorRef &v,
                                         const ConstMatrixRef &actMatrix,
                                         const context::RCM &constraint_model,
                                         context::RCD &constraint_data) {
  long nu = actMatrix.cols();
  long d = (long)constraint_model.size();
  context::VectorXs out(nu + d);
  underactuatedConstrainedInverseDynamics(
      model, data, q, v, actMatrix, constraint_model, constraint_data, out);

  return bp::make_tuple((context::VectorXs)out.head(nu),
                        (context::VectorXs)out.tail(d));
}
#endif

void exposePinocchioFunctions() {
  exposeFrameFunctions();
  exposeFlyHigh();
  exposeCenterOfMassFunctions();

#ifdef PROXDDP_PINOCCHIO_V3
  bp::def("underactuatedConstrainedInverseDynamics",
          underactuatedConstraintInvDyn_proxy,
          bp::args("model", "data", "q", "v", "actMatrix", "constraint_model",
                   "constraint_data"),
          "Compute the gravity-compensating torque for a pinocchio Model under "
          "a rigid constraint.");
#endif
}
} // namespace python
} // namespace proxddp

#endif
