#pragma once

#include "proxddp/modelling/dynamics/linear-ode.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>

namespace proxddp {
namespace dynamics {
template <typename Scalar>
void LinearODETpl<Scalar>::forward(const ConstVectorRef &x,
                                   const ConstVectorRef &u,
                                   ODEData &data) const {
  fmt::print("A:\n{}\n", A_);
  fmt::print("B:\n{}\n", B_);
  fmt::print("B * u:\n{}\n", VectorXs(B_ * u).transpose());
  data.xdot_ = A_ * x + B_ * u + c_;
}
template <typename Scalar>
void LinearODETpl<Scalar>::dForward(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    ODEData &data) const {
  return;
}
} // namespace dynamics
} // namespace proxddp
