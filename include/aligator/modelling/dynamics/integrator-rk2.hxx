/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-rk2.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
IntegratorRK2Tpl<Scalar>::IntegratorRK2Tpl(
    const xyz::polymorphic<ODEType> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics)
    , timestep_(timestep) {}

template <typename Scalar>
void IntegratorRK2Tpl<Scalar>::forward(const ConstVectorRef &x,
                                       const ConstVectorRef &u,
                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  ODEData &cd1 = static_cast<ODEData &>(*d.continuous_data);
  ODEData &cd2 = static_cast<ODEData &>(*d.continuous_data2);

  this->ode_->forward(x, u, cd1);
  Scalar dt_2_ = 0.5 * timestep_;
  d.dx1_ = dt_2_ * cd1.xdot_;
  this->space_next_->integrate(x, d.dx1_, d.x1_);

  this->ode_->forward(d.x1_, u, cd2);
  d.dx_ = timestep_ * cd2.xdot_;
  this->space_next_->integrate(x, d.dx_, d.xnext_);
}

template <typename Scalar>
void IntegratorRK2Tpl<Scalar>::dForward(const ConstVectorRef &x,
                                        const ConstVectorRef &u,
                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  ODEData &cd1 = static_cast<ODEData &>(*d.continuous_data);
  ODEData &cd2 = static_cast<ODEData &>(*d.continuous_data2);

  // x1 = x + dx1
  // dx1_dz = Transport(d(dx1)_dz) + dx_dz
  this->ode_->dForward(x, u, cd1);
  Scalar dt_2_ = 0.5 * timestep_;
  d.Jx() = dt_2_ * cd1.Jx();
  d.Ju() = dt_2_ * cd1.Ju();
  this->space_next_->JintegrateTransport(x, d.dx1_, d.Jx(), 1);
  this->space_next_->JintegrateTransport(x, d.dx1_, d.Ju(), 1);
  this->space_next_->Jintegrate(x, d.dx1_, d.Jtmp_xnext, 0);
  d.Jx() += d.Jtmp_xnext;

  // J = d(x+dx)_dz = d(x+dx)_dx1 * dx1_dz
  // then transport J to xnext = exp(dx) * x1
  this->ode_->dForward(d.x1_, u, cd2);
  d.Jx() = (timestep_ * cd2.Jx()) * d.Jx();
  d.Ju() = (timestep_ * cd2.Jx()) * d.Ju() + timestep_ * cd2.Ju();
  this->space_next_->JintegrateTransport(d.x1_, d.dx_, d.Jx(), 1);
  this->space_next_->JintegrateTransport(d.x1_, d.dx_, d.Ju(), 1);

  this->space_next_->Jintegrate(d.x1_, d.dx_, d.Jtmp_xnext, 0);
  d.Jx() += d.Jtmp_xnext;
}

} // namespace dynamics
} // namespace aligator
