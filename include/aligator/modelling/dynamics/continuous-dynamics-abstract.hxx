/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar>
ContinuousDynamicsAbstractTpl<Scalar>::ContinuousDynamicsAbstractTpl(
    ManifoldPtr space, const int nu)
    : space_(space), nu_(nu) {}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
ContinuousDynamicsAbstractTpl<Scalar>::createData() const {
  return std::make_shared<Data>(ndx(), nu());
}

template <typename Scalar>
ContinuousDynamicsDataTpl<Scalar>::ContinuousDynamicsDataTpl(const int ndx,
                                                             const int nu)
    : value_(ndx), Jx_(ndx, ndx), Ju_(ndx, nu), Jxdot_(ndx, ndx), xdot_(ndx) {
  value_.setZero();
  Jx_.setZero();
  Ju_.setZero();
  Jxdot_.setZero();
  // initialization for ODE models
  Jxdot_.diagonal().setConstant(Scalar(-1.));
  xdot_.setZero();
}

} // namespace dynamics
} // namespace aligator
