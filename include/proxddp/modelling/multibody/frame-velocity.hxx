#pragma once

#include "proxddp/modelling/multibody/frame-velocity.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace proxddp {

template <typename Scalar>
void FrameVelocityResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  auto q = x.head(model.nq);
  auto v = x.segment(model.nq, model.nv);
  pinocchio::forwardKinematics(model, d.pin_data_, q, v);
  d.value_ =
      (pinocchio::getFrameVelocity(model, d.pin_data_, pin_frame_id_, type_) -
       v_ref_)
          .toVector();
}

template <typename Scalar>
void FrameVelocityResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::getFrameVelocityDerivatives(model, d.pin_data_, pin_frame_id_,
                                         type_, d.Jx_.leftCols(model.nv),
                                         d.Jx_.rightCols(model.nv));
}

template <typename Scalar>
FrameVelocityDataTpl<Scalar>::FrameVelocityDataTpl(
    const FrameVelocityResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 6),
      pin_data_(*model->pin_model_) {}

} // namespace proxddp
