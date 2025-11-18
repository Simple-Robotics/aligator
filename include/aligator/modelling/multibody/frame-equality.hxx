#pragma once

#include "aligator/modelling/multibody/frame-equality.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace aligator {

template <typename Scalar>
void FrameEqualityResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  const ConstVectorRef q = x.head(pin_model_.nq);
  pinocchio::forwardKinematics(pin_model_, pdata, q);
  pinocchio::updateFramePlacement(pin_model_, pdata, pin_frame_id1_);
  pinocchio::updateFramePlacement(pin_model_, pdata, pin_frame_id2_);

  d.RMf2_ = pdata.oMf[pin_frame_id1_].act(f1MR_ref_).actInv(
      pdata.oMf[pin_frame_id2_]);
  d.value_ = pinocchio::log6(d.RMf2_).toVector();
}

template <typename Scalar>
void FrameEqualityResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::Jlog6(d.RMf2_, d.RJlog6f2_);
  pinocchio::computeJointJacobians(pin_model_, pdata);
  pinocchio::getFrameJacobian(pin_model_, pdata, pin_frame_id1_,
                              pinocchio::WORLD, d.wJf1_);
  pinocchio::getFrameJacobian(pin_model_, pdata, pin_frame_id2_,
                              pinocchio::WORLD, d.wJf2_);

  d.Jx_.leftCols(pin_model_.nv) =
      d.RJlog6f2_ *
      (pdata.oMf[pin_frame_id2_].act(f2MR_ref_)).toActionMatrixInverse() *
      (d.wJf2_ - d.wJf1_);
}

template <typename Scalar>
FrameEqualityDataTpl<Scalar>::FrameEqualityDataTpl(
    const FrameEqualityResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, 6)
    , pin_data_(model.pin_model_)
    , RJlog6f2_(6, 6)
    , wJf1_(6, model.pin_model_.nv)
    , wJf2_(6, model.pin_model_.nv) {
  wJf1_.setZero();
  wJf2_.setZero();
  RJlog6f2_.setZero();
}

} // namespace aligator
