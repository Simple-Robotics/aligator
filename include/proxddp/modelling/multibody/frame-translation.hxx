#pragma once

#include "proxddp/modelling/multibody/frame-translation.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace proxddp {

template <typename Scalar>
void FrameTranslationResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                   BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::forwardKinematics(model, pdata, x.head(model.nq));
  pinocchio::updateFramePlacement(model, pdata, pin_frame_id_);

  d.value_ = pdata.oMf[pin_frame_id_].translation() - p_ref_;
}

template <typename Scalar>
void FrameTranslationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &, BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::computeJointJacobians(model, pdata);
  pinocchio::getFrameJacobian(model, pdata, pin_frame_id_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.fJf_);
  d.Jx_.leftCols(model.nv) = d.fJf_.topRows(3);
}

template <typename Scalar>
FrameTranslationDataTpl<Scalar>::FrameTranslationDataTpl(
    const FrameTranslationResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, model.ndx2, 3), pin_data_(*model.pin_model_),
      fJf_(6, model.pin_model_->nv) {
  fJf_.setZero();
}

} // namespace proxddp
