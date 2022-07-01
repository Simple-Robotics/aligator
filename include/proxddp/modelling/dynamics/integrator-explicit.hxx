#pragma once

#include "proxddp/modelling/dynamics/integrator-explicit.hpp"

namespace proxddp {
namespace dynamics {
template <typename Scalar>
ExplicitIntegratorAbstractTpl<Scalar>::ExplicitIntegratorAbstractTpl(
    const shared_ptr<ODEType> &cont_dynamics)
    : Base(cont_dynamics->space_, cont_dynamics->nu()), ode_(cont_dynamics) {}

template <typename Scalar>
shared_ptr<DynamicsDataTpl<Scalar>>
ExplicitIntegratorAbstractTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this);
}

template <typename Scalar>
ExplicitIntegratorDataTpl<Scalar>::ExplicitIntegratorDataTpl(
    const ExplicitIntegratorAbstractTpl<Scalar> *integrator)
    : Base(integrator->ndx1, integrator->nu, integrator->out_space()) {
  continuous_data = std::static_pointer_cast<ODEDataTpl<Scalar>>(
      integrator->ode_->createData());
}
} // namespace dynamics
} // namespace proxddp
