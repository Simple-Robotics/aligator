#pragma once

#include "aligator/modelling/multibody/frame-placement.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace aligator {

template <typename Scalar>
void FramePlacementResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::forwardKinematics(model, pdata, x.head(model.nq));
  pinocchio::updateFramePlacement(model, pdata, pin_frame_id_);

  d.rMf_ = p_ref_inverse_ * pdata.oMf[pin_frame_id_];
  d.value_ = pinocchio::log6(d.rMf_).toVector();
}

template <typename Scalar>
void FramePlacementResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                         BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::Jlog6(d.rMf_, d.rJf_);
  pinocchio::computeJointJacobians(model, pdata);
  pinocchio::getFrameJacobian(model, pdata, pin_frame_id_, pinocchio::LOCAL,
                              d.fJf_);
  d.Jx_.leftCols(model.nv) = d.rJf_ * d.fJf_;
}

template <typename Scalar>
FramePlacementDataTpl<Scalar>::FramePlacementDataTpl(
    const FramePlacementResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, model.ndx2, 6), pin_data_(*model.pin_model_),
      rJf_(6, 6), fJf_(6, model.pin_model_->nv) {
  rJf_.setZero();
  fJf_.setZero();
}

} // namespace aligator
