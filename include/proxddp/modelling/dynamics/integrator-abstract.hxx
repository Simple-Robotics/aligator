#pragma once

#include "proxddp/modelling/dynamics/integrator-abstract.hpp"

namespace proxddp {
namespace dynamics {
template <typename Scalar>
IntegratorAbstractTpl<Scalar>::IntegratorAbstractTpl(
    const shared_ptr<ContinuousDynamics> &cont_dynamics)
    : Base(cont_dynamics->space_, cont_dynamics->nu()),
      continuous_dynamics_(cont_dynamics) {}

template <typename Scalar>
shared_ptr<StageFunctionDataTpl<Scalar>>
IntegratorAbstractTpl<Scalar>::createData() const {
  return std::make_shared<IntegratorDataTpl<Scalar>>(this);
}

template <typename Scalar>
IntegratorDataTpl<Scalar>::IntegratorDataTpl(
    const IntegratorAbstractTpl<Scalar> *integrator)
    : Base(integrator->ndx1, integrator->nu, integrator->ndx2,
           integrator->ndx2),
      continuous_data(integrator->continuous_dynamics_->createData()),
      xdot_(integrator->continuous_dynamics_->ndx()) {
  xdot_.setZero();
}

} // namespace dynamics
} // namespace proxddp
