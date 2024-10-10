#pragma once

#include "aligator/modelling/multibody/frame-translation.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace aligator {
template <typename Scalar>
FrameTranslationResidualTpl<Scalar>::FrameTranslationResidualTpl(
    const int ndx, const int nu, const Model &model,
    const Vector3s &frame_trans, const pinocchio::FrameIndex frame_id)
    : Base(ndx, nu, 3), pin_model_(model), p_ref_(frame_trans) {
  pin_frame_id_ = frame_id;
}

template <typename Scalar>
void FrameTranslationResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                   BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::forwardKinematics(pin_model_, pdata, x.head(pin_model_.nq));
  pinocchio::updateFramePlacement(pin_model_, pdata, pin_frame_id_);

  d.value_ = pdata.oMf[pin_frame_id_].translation() - p_ref_;
}

template <typename Scalar>
void FrameTranslationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::computeJointJacobians(pin_model_, pdata);
  pinocchio::getFrameJacobian(pin_model_, pdata, pin_frame_id_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.fJf_);
  d.Jx_.leftCols(pin_model_.nv) = d.fJf_.topRows(3);
}

template <typename Scalar>
FrameTranslationDataTpl<Scalar>::FrameTranslationDataTpl(
    const FrameTranslationResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, 3), pin_data_(model.pin_model_),
      fJf_(6, model.pin_model_.nv) {
  fJf_.setZero();
}

} // namespace aligator
