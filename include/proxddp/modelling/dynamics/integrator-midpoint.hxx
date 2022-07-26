#pragma once

#include "proxddp/modelling/dynamics/integrator-midpoint.hpp"
#include "proxddp/utils/exceptions.hpp"

namespace proxddp {
namespace dynamics {

template <typename Scalar>
IntegratorMidpointTpl<Scalar>::IntegratorMidpointTpl(
    const shared_ptr<ContinuousDynamics> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics), timestep_(timestep) {
  if (timestep <= 0.) {
    proxddp_runtime_error("Timestep must be positive!");
  }
}

template <typename Scalar>
void IntegratorMidpointTpl<Scalar>::evaluate(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    DynamicsDataTpl<Scalar> &data) const {
  IntegratorMidpointDataTpl<Scalar> &d =
      static_cast<IntegratorMidpointDataTpl<Scalar> &>(data);
  const ContinuousDynamics *contdyn = this->continuous_dynamics_.get();
  const Manifold &space = contdyn->space();
  ContinuousDynamicsDataTpl<Scalar> *contdata = d.continuous_data.get();
  // define xdot = (y-x) / timestep
  space.difference(x, y, d.xdot_);
  d.xdot_ /= timestep_;
  // define x1 = midpoint of x,y
  space.interpolate(x, y, 0.5, d.x1_);
  space.difference(x, d.x1_, d.dx1_);
  // evaluate on (x1, u, xdot)
  contdyn->evaluate(d.x1_, u, d.xdot_, *contdata);
  d.value_ = contdata->value_;
}

template <typename Scalar>
void IntegratorMidpointTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    DynamicsDataTpl<Scalar> &data) const {
  IntegratorMidpointDataTpl<Scalar> &d =
      static_cast<IntegratorMidpointDataTpl<Scalar> &>(data);
  const ContinuousDynamics *contdyn = this->continuous_dynamics_.get();
  const Manifold &space = contdyn->space();
  ContinuousDynamicsDataTpl<Scalar> *contdata = d.continuous_data.get();

  auto dx = d.dx1_ * 2;
  // jacobians of xdot estimate
  space.Jdifference(x, y, d.J_v_0, 0);
  space.Jdifference(x, y, d.J_v_1, 1);
  d.J_v_0 /= timestep_;
  d.J_v_1 /= timestep_;
  // d.x1_ contains midpoint of x,y
  // compute jacobians
  contdyn->computeJacobians(d.x1_, u, d.xdot_, *contdata);

  // bring the Jacobian in arg1 from xmid to x
  space.JintegrateTransport(x, d.dx1_, contdata->Jx_, 1);
  data.Jx_ = 0.5 * contdata->Jx_ + contdata->Jxdot_ * d.J_v_0;

  data.Ju_ = contdata->Ju_;

  // bring the Jacobian in x = y - dx to y
  space.JintegrateTransport(y, -dx, contdata->Jx_, 1);
  data.Jy_ = 0.5 * contdata->Jx_ + contdata->Jxdot_ * d.J_v_1;
}

template <typename Scalar>
shared_ptr<DynamicsDataTpl<Scalar>>
IntegratorMidpointTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this);
}

} // namespace dynamics
} // namespace proxddp
