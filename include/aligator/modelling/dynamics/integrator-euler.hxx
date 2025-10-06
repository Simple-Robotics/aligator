/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-euler.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
IntegratorEulerTpl<Scalar>::IntegratorEulerTpl(
    const xyz::polymorphic<ODEType> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics)
    , timestep_(timestep) {}

template <typename Scalar>
void IntegratorEulerTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u,
    ExplicitDynamicsDataTpl<Scalar> &data) const {
  Data &d = static_cast<Data &>(data);
  ODEData &cdata = *d.continuous_data;
  this->ode_->forward(x, u, cdata);
  d.dx_ = timestep_ * cdata.xdot_;
  this->space_next().integrate(x, d.dx_, d.xnext_);
}

template <typename Scalar>
void IntegratorEulerTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &u,
    ExplicitDynamicsDataTpl<Scalar> &data) const {
  Data &d = static_cast<Data &>(data);
  ODEData &cdata = *d.continuous_data;

  // d(dx)_z = dt * df_dz
  // then transport to x+dx
  this->ode_->dForward(x, u, cdata);
  d.Jx() = timestep_ * cdata.Jx(); // ddx_dx
  d.Ju() = timestep_ * cdata.Ju(); // ddx_du
  this->space_next().JintegrateTransport(x, d.dx_, d.Jx(), 1);
  this->space_next().Jintegrate(x, d.dx_, d.Jtmp_xnext, 0);
  d.Jx() += d.Jtmp_xnext;
  this->space_next().JintegrateTransport(x, d.dx_, d.Ju(), 1);
}
} // namespace dynamics
} // namespace aligator
