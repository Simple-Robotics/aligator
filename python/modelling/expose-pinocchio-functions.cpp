#include "proxddp/fwd.hpp"
#include "proxddp/python/functions.hpp"

#ifdef PROXDDP_WITH_PINOCCHIO
#include "proxddp/modelling/multibody/frame-placement.hpp"
#include "proxddp/modelling/multibody/frame-velocity.hpp"
#include "proxddp/modelling/multibody/frame-translation.hpp"

namespace proxddp {
namespace python {

void exposePinocchioFunctions() {
  using context::Manifold;
  using context::Scalar;
  using context::StageFunction;
  using Model = pinocchio::ModelTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Motion = pinocchio::MotionTpl<Scalar>;

  using FramePlacement = FramePlacementResidualTpl<Scalar>;
  using FramePlacementData = FramePlacementDataTpl<Scalar>;

  using FrameVelocity = FrameVelocityResidualTpl<Scalar>;
  using FrameVelocityData = FrameVelocityDataTpl<Scalar>;

  using FrameTranslation = FrameTranslationResidualTpl<Scalar>;
  using FrameTranslationData = FrameTranslationDataTpl<Scalar>;

  bp::register_ptr_to_python<shared_ptr<PinData>>();

  bp::class_<FramePlacement, bp::bases<StageFunction>>(
      "FramePlacementResidual", "Frame placement residual function.",
      bp::init<int, int, shared_ptr<Model>, const SE3 &, pinocchio::FrameIndex>(
          bp::args("self", "ndx", "nu", "model", "p_ref", "id")))
      .add_property("frame_id", &FramePlacement::getFrameId,
                    &FramePlacement::setFrameId)
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

  bp::class_<FrameVelocity, bp::bases<StageFunction>>(
      "FrameVelocityResidual", "Frame velocity residual function.",
      bp::init<int, int, shared_ptr<Model>, const Motion &,
               pinocchio::FrameIndex, pinocchio::ReferenceFrame>(bp::args(
          "self", "ndx", "nu", "model", "v_ref", "id", "reference_frame")))
      .add_property("frame_id", &FrameVelocity::getFrameId,
                    &FrameVelocity::setFrameId)
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

  bp::class_<FrameTranslation, bp::bases<StageFunction>>(
      "FrameTranslationResidual", "Frame placement residual function.",
      bp::init<int, int, shared_ptr<Model>, const context::VectorXs &,
               pinocchio::FrameIndex>(
          bp::args("self", "ndx", "nu", "model", "p_ref", "id")))
      .add_property("frame_id", &FrameTranslation::getFrameId,
                    &FrameTranslation::setFrameId)
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

} // namespace python
} // namespace proxddp

#endif
