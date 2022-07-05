#pragma once

#include "proxddp/modelling/dynamics/integrator-midpoint.hpp"

namespace proxddp {
namespace dynamics {

template <typename Scalar>
IntegratorMidpointTpl<Scalar>::IntegratorMidpointTpl(
    const shared_ptr<ContinuousDynamics> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics), timestep_(timestep) {
  if (timestep <= 0.) {
    throw std::domain_error("Timestep must be positive!");
  }
}

} // namespace dynamics
} // namespace proxddp
