/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-abstract.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
IntegratorAbstractTpl<Scalar>::IntegratorAbstractTpl(
    const xyz::polymorphic<ContinuousDynamics> &cont_dynamics)
    : Base(cont_dynamics->space_, cont_dynamics->nu()),
      continuous_dynamics_(cont_dynamics) {}

template <typename Scalar>
auto IntegratorAbstractTpl<Scalar>::createData() const -> shared_ptr<BaseData> {
  return std::make_shared<IntegratorDataTpl<Scalar>>(*this);
}

template <typename Scalar>
IntegratorDataTpl<Scalar>::IntegratorDataTpl(
    const IntegratorAbstractTpl<Scalar> &model)
    : Base(model), continuous_data(model.continuous_dynamics_->createData()),
      xdot_(model.continuous_dynamics_->ndx()) {
  xdot_.setZero();
}

} // namespace dynamics
} // namespace aligator
