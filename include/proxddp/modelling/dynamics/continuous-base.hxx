#pragma once

#include "proxddp/modelling/dynamics/continuous-base.hpp"

namespace proxddp {
namespace dynamics {
template <typename Scalar>
ContinuousDynamicsAbstractTpl<Scalar>::ContinuousDynamicsAbstractTpl(
    const ManifoldPtr &space, const int nu)
    : space_(space), nu_(nu) {}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
ContinuousDynamicsAbstractTpl<Scalar>::createData() const {
  return std::make_shared<Data>(ndx(), nu());
}

template <typename Scalar>
ContinuousDynamicsDataTpl<Scalar>::ContinuousDynamicsDataTpl(const int ndx,
                                                             const int nu)
    : value_(ndx), Jx_(ndx, ndx), Ju_(ndx, nu), Jxdot_(ndx, ndx) {
  value_.setZero();
  Jx_.setZero();
  Ju_.setZero();
  Jxdot_.setZero();
}

} // namespace dynamics
} // namespace proxddp
