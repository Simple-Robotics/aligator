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
shared_ptr<DynamicsDataTpl<Scalar>>
IntegratorMidpointTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this);
}

} // namespace dynamics
} // namespace proxddp
