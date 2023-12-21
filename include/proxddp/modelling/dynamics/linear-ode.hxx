#pragma once

#include "proxddp/modelling/dynamics/linear-ode.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
void LinearODETpl<Scalar>::forward(const ConstVectorRef &x,
                                   const ConstVectorRef &u,
                                   ODEData &data) const {
  data.xdot_ = A_ * x + B_ * u + c_;
}
template <typename Scalar>
void LinearODETpl<Scalar>::dForward(const ConstVectorRef &,
                                    const ConstVectorRef &, ODEData &) const {
  return;
}
} // namespace dynamics
} // namespace aligator
