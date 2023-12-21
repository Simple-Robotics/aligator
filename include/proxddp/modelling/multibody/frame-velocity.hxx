#pragma once

#include "proxddp/modelling/multibody/frame-velocity.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace aligator {

template <typename Scalar>
FrameVelocityResidualTpl<Scalar>::FrameVelocityResidualTpl(
    const int ndx, const int nu, const shared_ptr<Model> &model,
    const Motion &velocity, const pinocchio::FrameIndex frame_id,
    const pinocchio::ReferenceFrame type)
    : Base(ndx, nu, 6), pin_model_(model), vref_(velocity), type_(type) {
  pin_frame_id_ = frame_id;
  if (ndx < model->nv * 2) {
    PROXDDP_RUNTIME_ERROR("Specified manifold dimension is incompatible. It "
                          "needs to be at least 2 * nv.");
  }
}

template <typename Scalar>
void FrameVelocityResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  auto q = x.head(model.nq);
  auto v = x.segment(model.nq, model.nv);
  pinocchio::forwardKinematics(model, d.pin_data_, q, v);
  pinocchio::updateFramePlacement(model, d.pin_data_, pin_frame_id_);
  d.value_ =
      (pinocchio::getFrameVelocity(model, d.pin_data_, pin_frame_id_, type_) -
       vref_)
          .toVector();
}

template <typename Scalar>
void FrameVelocityResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  auto q = x.head(model.nq);
  auto v = x.segment(model.nq, model.nv);
  VectorXs a = VectorXs::Zero(model.nv);
  pinocchio::computeForwardKinematicsDerivatives(model, d.pin_data_, q, v, a);
  pinocchio::getFrameVelocityDerivatives(model, d.pin_data_, pin_frame_id_,
                                         type_, d.Jx_.leftCols(model.nv),
                                         d.Jx_.rightCols(model.nv));
}

template <typename Scalar>
FrameVelocityDataTpl<Scalar>::FrameVelocityDataTpl(
    const FrameVelocityResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, model.ndx2, 6), pin_data_(*model.pin_model_) {}

} // namespace aligator
