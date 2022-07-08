#pragma once

#include "proxddp/modelling/multibody/frame-velocity.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace proxddp {

template <typename Scalar>
void FrameVelocityResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                 const ConstVectorRef &u,
                                                 const ConstVectorRef &y,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::forwardKinematics(model, pdata, x.head(model.nq));
  d.value_ = (pinocchio::getFrameVelocity(model, pdata, pin_frame_id_, type_) - v_ref_).toVector();
}

template <typename Scalar>
void FrameVelocityResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::getFrameVelocityDerivatives(model, pdata, pin_frame_id_, type_, d.Jx_.leftCols(model.nv), d.Jx_.rightCols(model.nv));
}

template <typename Scalar>
FrameVelocityDataTpl<Scalar>::FrameVelocityDataTpl(
    const FrameVelocityResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 6),
      pin_data_(*model->pin_model_){}

} // namespace proxddp
