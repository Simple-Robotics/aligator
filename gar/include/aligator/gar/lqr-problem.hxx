#pragma once

#include "lqr-problem.hpp"

namespace aligator::gar {

template <typename Scalar>
LQRKnotTpl<Scalar>::LQRKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth)
    : nx(nx), nu(nu), nc(nc), nx2(nx2), nth(nth),    //
      Q(nx, nx), S(nx, nu), R(nu, nu), q(nx), r(nu), //
      A(nx2, nx), B(nx2, nu), E(nx2, nx), f(nx2),    //
      C(nc, nx), D(nc, nu), d(nc), Gth(nth, nth), Gx(nx, nth), Gu(nu, nth),
      Gv(nc, nth), gamma(nth) {
  Q.setZero();
  S.setZero();
  R.setZero();
  q.setZero();
  r.setZero();

  A.setZero();
  B.setZero();
  E.setZero();
  f.setZero();

  C.setZero();
  D.setZero();
  d.setZero();

  Gth.setZero();
  Gx.setZero();
  Gu.setZero();
  Gv.setZero();
  gamma.setZero();
}

template <typename Scalar>
Scalar LQRProblemTpl<Scalar>::evaluate(
    const VectorOfVectors &xs, const VectorOfVectors &us,
    const std::optional<ConstVectorRef> &theta_) const {
  if ((int)xs.size() != horizon() + 1)
    return 0.;
  if ((int)us.size() < horizon())
    return 0.;

  if (!isInitialized())
    return 0.;

  Scalar ret = 0.;
  for (uint i = 0; i <= (uint)horizon(); i++) {
    const LQRKnotTpl<Scalar> &knot = stages[i];
    ret += 0.5 * xs[i].dot(knot.Q * xs[i]) + xs[i].dot(knot.q);
    if (i == (uint)horizon())
      break;
    ret += 0.5 * us[i].dot(knot.R * us[i]) + us[i].dot(knot.r);
    ret += xs[i].dot(knot.S * us[i]);
  }

  if (!isParameterized())
    return ret;

  if (theta_.has_value()) {
    ConstVectorRef th = theta_.value();
    for (uint i = 0; i <= (uint)horizon(); i++) {
      const LQRKnotTpl<Scalar> &knot = stages[i];
      ret += 0.5 * th.dot(knot.Gth * th);
      ret += th.dot(knot.Gx.transpose() * xs[i]);
      ret += th.dot(knot.gamma);
      if (i == (uint)horizon())
        break;
      ret += th.dot(knot.Gu.transpose() * us[i]);
    }
  }

  return ret;
}

} // namespace aligator::gar
