#pragma once

#include "proxddp/modelling/multibody/fly-high.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>

namespace proxddp {

template <typename Scalar>
FlyHighResidualTpl<Scalar>::FlyHighResidualTpl(
    shared_ptr<PhaseSpace> state, const pinocchio::FrameIndex frame_id,
    Scalar slope, int nu)
    : Base(state->ndx(), nu, NR), slope_(slope), pmodel_(state->getModel()) {
  pin_frame_id_ = frame_id;
}

template <typename Scalar>
void FlyHighResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                          BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  auto q = x.head(pmodel_.nq);
  auto v = x.segment(pmodel_.nq, pmodel_.nv);
  pinocchio::forwardKinematics(pmodel_, d.pdata_, q, v);
  pinocchio::updateFramePlacement(pmodel_, d.pdata_, pin_frame_id_);

  d.value_ = pinocchio::getFrameVelocity(pmodel_, d.pdata_, pin_frame_id_,
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
  const int nv = pmodel_.nv;
  auto q = x.head(pmodel_.nq);
  auto v = x.segment(pmodel_.nq, nv);
  auto a = VectorXs::Zero(nv);

  pinocchio::computeForwardKinematicsDerivatives(pmodel_, d.pdata_, q, v, a);
  pinocchio::getFrameVelocityDerivatives(pmodel_, d.pdata_, pin_frame_id_,
                                         pinocchio::LOCAL, d.l_dnu_dq,
                                         d.l_dnu_dv);
  const Vector3s &vf = pinocchio::getFrameVelocity(
                           pmodel_, d.pdata_, pin_frame_id_, pinocchio::LOCAL)
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

} // namespace proxddp
