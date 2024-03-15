/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-midpoint.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
IntegratorMidpointTpl<Scalar>::IntegratorMidpointTpl(
    const shared_ptr<ContinuousDynamics> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics), timestep_(timestep) {
  if (timestep <= 0.) {
    ALIGATOR_RUNTIME_ERROR("Timestep must be positive!");
  }
}

template <typename Scalar>
void IntegratorMidpointTpl<Scalar>::evaluate(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    DynamicsDataTpl<Scalar> &data) const {
  IntegratorMidpointDataTpl<Scalar> &d =
      static_cast<IntegratorMidpointDataTpl<Scalar> &>(data);
  const ContinuousDynamics &contdyn = *this->continuous_dynamics_;
  const Manifold &space = contdyn.space();
  auto &contdata = d.continuous_data;
  // define xdot = (y-x) / timestep
  space.difference(x, y, d.xdot_);
  d.dx1_ = d.xdot_ * 0.5;
  d.xdot_ /= timestep_;
  // define x1 = midpoint of x,y
  space.integrate(x, d.dx1_, d.x1_);
  // evaluate on (x1, u, xdot)
  d.common_models.evaluate(d.x1_, u, d.common_datas);
  contdyn.evaluate(d.x1_, u, d.xdot_, *contdata);
  d.value_ = contdata->value_ * timestep_;
}

template <typename Scalar>
void IntegratorMidpointTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    DynamicsDataTpl<Scalar> &data) const {
  IntegratorMidpointDataTpl<Scalar> &d =
      static_cast<IntegratorMidpointDataTpl<Scalar> &>(data);
  const ContinuousDynamics &contdyn = *this->continuous_dynamics_;
  const Manifold &space = contdyn.space();
  auto &contdata = d.continuous_data;

  auto dx = d.dx1_ * 2;
  // jacobians of xdot estimate
  space.Jdifference(x, y, d.J_v_0, 0);
  space.Jdifference(x, y, d.J_v_1, 1);

  auto &Jtm0 = d.Jtm0;
  auto &Jtm1 = d.Jtm1;
  space.Jintegrate(x, d.dx1_, Jtm0, 0);
  space.Jintegrate(x, d.dx1_, Jtm1, 1);
  Jtm0 = Jtm0 + 0.5 * Jtm1 * d.J_v_0;
  Jtm1 = 0.5 * Jtm1 * d.J_v_1;

  // d.x1_ contains midpoint of x,y
  // compute jacobians
  d.common_models.computeGradients(d.x1_, u, d.common_datas);
  contdyn.computeJacobians(d.x1_, u, d.xdot_, *contdata);
  // bring the Jacobian in arg1 from xmid to x
  space.JintegrateTransport(x, d.dx1_, contdata->Jx_, 1);
  data.Jx_ = contdata->Jx_ * Jtm0 * timestep_ + contdata->Jxdot_ * d.J_v_0;
  data.Ju_ = contdata->Ju_ * timestep_;
  // bring the Jacobian in x = y - dx to y
  space.JintegrateTransport(y, -dx, contdata->Jx_, 1);
  data.Jy_ = contdata->Jx_ * Jtm1 * timestep_ + contdata->Jxdot_ * d.J_v_1;
}

template <typename Scalar>
shared_ptr<DynamicsDataTpl<Scalar>>
IntegratorMidpointTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this);
}

template <typename Scalar>
shared_ptr<DynamicsDataTpl<Scalar>> IntegratorMidpointTpl<Scalar>::createData(
    const CommonModelDataContainer &container) const {
  return std::make_shared<Data>(this, container);
}

} // namespace dynamics
} // namespace aligator
