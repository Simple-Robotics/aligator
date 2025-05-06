/// @file ode-abstract.hxx  Implement the ContinuousDynamicsAbstractTpl
/// interface for BaseODETpl.
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

#include "aligator/core/manifold-base.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
void ODEAbstractTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                      const ConstVectorRef &u,
                                      const ConstVectorRef &xdot,
                                      Data &data) const {
  Data &d = static_cast<Data &>(data);
  this->forward(x, u, d);
  d.value_ = d.xdot_ - xdot;
}

template <typename Scalar>
void ODEAbstractTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                              const ConstVectorRef &u,
                                              const ConstVectorRef &,
                                              Data &data) const {
  Data &d = static_cast<Data &>(data);
  this->dForward(x, u, d);
  data.Jxdot_.setZero();
  data.Jxdot_.diagonal().setConstant(Scalar(-1.));
}

} // namespace dynamics
} // namespace aligator
