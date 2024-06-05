/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-rk2.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
IntegratorRK2Tpl<Scalar>::IntegratorRK2Tpl(
    const shared_ptr<ODEType> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics), timestep_(timestep) {}

template <typename Scalar>
void IntegratorRK2Tpl<Scalar>::forward(const ConstVectorRef &x,
                                       const ConstVectorRef &u,
                                       BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  using ODEData = ODEDataTpl<Scalar>;
  ODEData &cd1 = static_cast<ODEData &>(*d.continuous_data);
  ODEData &cd2 = static_cast<ODEData &>(*d.continuous_data2);

  this->ode_->forward(x, u, cd1);
  d.dx1_ = dt_2_ * cd1.xdot_;
  this->space_next_->integrate(x, d.dx1_, d.x1_);

  d.common_models.evaluate(d.x1_, u, d.common_datas);
  this->ode_->forward(d.x1_, u, cd2);
  d.dx_ = timestep_ * cd2.xdot_;
  this->space_next_->integrate(x, d.dx_, d.xnext_);
}

template <typename Scalar>
void IntegratorRK2Tpl<Scalar>::dForward(const ConstVectorRef &x,
                                        const ConstVectorRef &u,
                                        BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  using ODEData = ODEDataTpl<Scalar>;
  ODEData &cd1 = static_cast<ODEData &>(*d.continuous_data);
  ODEData &cd2 = static_cast<ODEData &>(*d.continuous_data2);

  // x1 = x + dx1
  // dx1_dz = Transport(d(dx1)_dz) + dx_dz
  this->ode_->dForward(x, u, cd1);
  d.Jx_ = dt_2_ * cd1.Jx_;
  d.Ju_ = dt_2_ * cd1.Ju_;
  this->space_next_->JintegrateTransport(x, d.dx1_, d.Jx_, 1);
  this->space_next_->JintegrateTransport(x, d.dx1_, d.Ju_, 1);
  this->space_next_->Jintegrate(x, d.dx1_, d.Jtmp_xnext, 0);
  d.Jx_ += d.Jtmp_xnext;

  // J = d(x+dx)_dz = d(x+dx)_dx1 * dx1_dz
  // then transport J to xnext = exp(dx) * x1
  d.common_models.computeGradients(d.x1_, u, d.common_datas);
  this->ode_->dForward(d.x1_, u, cd2);
  d.Jx_ = (timestep_ * cd2.Jx_) * d.Jx_;
  d.Ju_ = (timestep_ * cd2.Jx_) * d.Ju_ + timestep_ * cd2.Ju_;
  this->space_next_->JintegrateTransport(d.x1_, d.dx_, d.Jx_, 1);
  this->space_next_->JintegrateTransport(d.x1_, d.dx_, d.Ju_, 1);

  this->space_next_->Jintegrate(d.x1_, d.dx_, d.Jtmp_xnext, 0);
  d.Jx_ += d.Jtmp_xnext;
}

template <typename Scalar>
IntegratorRK2DataTpl<Scalar>::IntegratorRK2DataTpl(
    const IntegratorRK2Tpl<Scalar> *integrator)
    : Base(integrator), x1_(integrator->space_next().neutral()) {
  CommonModelBuilderContainer model_builders;
  integrator->ode_->configure(model_builders);
  common_models = model_builders.createCommonModelContainer();
  common_datas = common_models.createData();
  continuous_data2 = std::static_pointer_cast<ODEData>(
      integrator->ode_->createData(common_datas));
}

template <typename Scalar>
IntegratorRK2DataTpl<Scalar>::IntegratorRK2DataTpl(
    const IntegratorRK2Tpl<Scalar> *integrator,
    const CommonModelDataContainer &container)
    : Base(integrator, container), x1_(integrator->space_next().neutral()) {
  CommonModelBuilderContainer model_builders;
  integrator->ode_->configure(model_builders);
  common_models = model_builders.createCommonModelContainer();
  common_datas = common_models.createData();
  continuous_data2 = std::static_pointer_cast<ODEData>(
      integrator->ode_->createData(common_datas));
}

} // namespace dynamics
} // namespace aligator
