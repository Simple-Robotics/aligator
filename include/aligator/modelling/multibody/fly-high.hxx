#pragma once

#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/multibody/fly-high.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace aligator {

template <typename Scalar>
FlyHighResidualTpl<Scalar>::FlyHighResidualTpl(
    const int ndx, const Model &model, const pinocchio::FrameIndex frame_id,
    Scalar slope, int nu)
    : Base(ndx, nu, NR), slope_(slope), pin_model_(model) {
  pin_frame_id_ = frame_id;
}

template <typename Scalar>
void FlyHighResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                          BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const ConstVectorRef q = x.head(pin_model_.nq);
  const ConstVectorRef v = x.segment(pin_model_.nq, pin_model_.nv);
  pinocchio::forwardKinematics(pin_model_, d.pdata_, q, v);
  pinocchio::updateFramePlacement(pin_model_, d.pdata_, pin_frame_id_);

  d.value_ = pinocchio::getFrameVelocity(pin_model_, d.pdata_, pin_frame_id_,
                                         pinocchio::LOCAL_WORLD_ALIGNED)
                 .linear()
                 .template head<2>();

  const Vector3s &tf = d.pdata_.oMf[pin_frame_id_].translation();
  d.ez = std::exp(-tf[2] * slope_);
  d.value_ *= d.ez;
}

template <typename Scalar>
void FlyHighResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const int nv = pin_model_.nv;
  const ConstVectorRef q = x.head(pin_model_.nq);
  const ConstVectorRef v = x.segment(pin_model_.nq, nv);
  VectorXs a = VectorXs::Zero(nv);

  pinocchio::computeForwardKinematicsDerivatives(pin_model_, d.pdata_, q, v,
                                                 pinocchio::make_const_ref(a));
  pinocchio::getFrameVelocityDerivatives(pin_model_, d.pdata_, pin_frame_id_,
                                         pinocchio::LOCAL, d.l_dnu_dq,
                                         d.l_dnu_dv);
  const Vector3s &vf =
      pinocchio::getFrameVelocity(pin_model_, d.pdata_, pin_frame_id_,
                                  pinocchio::LOCAL)
          .linear();
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;
  const Matrix3s &R = d.pdata_.oMf[pin_frame_id_].rotation();

  d.vxJ.noalias() = pinocchio::skew(-vf) * d.l_dnu_dv.template bottomRows<3>();
  d.vxJ += d.l_dnu_dq.template topRows<3>();
  d.o_dv_dq.noalias() = R * d.vxJ;
  d.o_dv_dv.noalias() = R * d.l_dnu_dv.template topRows<3>();

  // First term: derivative of v_j
  d.Jx_.leftCols(nv) = d.o_dv_dq.template topRows<2>();
  d.Jx_.rightCols(nv) = d.o_dv_dv.template topRows<2>();
  d.Jx_ *= d.ez;

  // Second term: derivative of z
  d.Jx_.leftCols(nv).row(0) -= data.value_[0] * slope_ * d.o_dv_dv.row(2);
  d.Jx_.leftCols(nv).row(1) -= data.value_[1] * slope_ * d.o_dv_dv.row(2);
}

} // namespace aligator
#endif
