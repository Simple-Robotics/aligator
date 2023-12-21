#pragma once

#include "aligator/modelling/dynamics/integrator-euler.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
IntegratorEulerTpl<Scalar>::IntegratorEulerTpl(
    const shared_ptr<ODEType> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics), timestep_(timestep) {}

template <typename Scalar>
void IntegratorEulerTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u,
    ExplicitDynamicsDataTpl<Scalar> &data) const {
  Data &d = static_cast<Data &>(data);
  ODEDataTpl<Scalar> &cdata =
      static_cast<ODEDataTpl<Scalar> &>(*d.continuous_data);
  this->ode_->forward(x, u, cdata);
  d.dx_ = timestep_ * cdata.xdot_;
  this->space_next().integrate(x, d.dx_, d.xnext_);
}

template <typename Scalar>
void IntegratorEulerTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &u,
    ExplicitDynamicsDataTpl<Scalar> &data) const {
  Data &d = static_cast<Data &>(data);
  ODEDataTpl<Scalar> &cdata =
      static_cast<ODEDataTpl<Scalar> &>(*d.continuous_data);

  // d(dx)_z = dt * df_dz
  // then transport to x+dx
  this->ode_->dForward(x, u, cdata);
  d.Jx_ = timestep_ * cdata.Jx_; // ddx_dx
  d.Ju_ = timestep_ * cdata.Ju_; // ddx_du
  this->space_next().JintegrateTransport(x, d.dx_, d.Jx_, 1);
  this->space_next().JintegrateTransport(x, d.dx_, d.Ju_, 1);

  this->space_next().Jintegrate(x, d.dx_, d.Jtmp_xnext, 0);
  d.Jx_ += d.Jtmp_xnext;
}
} // namespace dynamics
} // namespace aligator
