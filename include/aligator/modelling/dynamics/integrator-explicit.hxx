/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-explicit.hpp"

namespace aligator {
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
shared_ptr<DynamicsDataTpl<Scalar>>
ExplicitIntegratorAbstractTpl<Scalar>::createData(
    const CommonModelDataContainer &container) const {
  return std::make_shared<Data>(this, container);
}

template <typename Scalar>
ExplicitIntegratorDataTpl<Scalar>::ExplicitIntegratorDataTpl(
    const ExplicitIntegratorAbstractTpl<Scalar> *integrator)
    : Base(integrator->ndx1, integrator->nu, integrator->nx2(),
           integrator->ndx2) {
  continuous_data =
      std::static_pointer_cast<ODEData>(integrator->ode_->createData());
}

template <typename Scalar>
ExplicitIntegratorDataTpl<Scalar>::ExplicitIntegratorDataTpl(
    const ExplicitIntegratorAbstractTpl<Scalar> *integrator,
    const CommonModelDataContainer &container)
    : Base(integrator->ndx1, integrator->nu, integrator->nx2(),
           integrator->ndx2) {
  continuous_data = std::static_pointer_cast<ODEData>(
      integrator->ode_->createData(container));
}

} // namespace dynamics
} // namespace aligator
