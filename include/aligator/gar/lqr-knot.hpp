/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"

namespace aligator {

template <typename Scalar> struct LQRKnot {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  uint nx, nu, nc;
  MatrixXs Q, S, R;
  VectorXs q, r;
  MatrixXs A, B, E;
  VectorXs f;
  MatrixXs C, D;
  VectorXs d;

  LQRKnot(uint nx, uint nu, uint nc)
      : nx(nx), nu(nu), nc(nc),                        //
        Q(nx, nx), S(nx, nu), R(nu, nu), q(nx), r(nu), //
        A(nx, nx), B(nx, nu), E(nx, nx), f(nx),        //
        C(nc, nx), D(nc, nu), d(nc) {
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
  }
};

template <typename T> struct LQRProblem {
  // last stage should have nu = 0
  std::vector<LQRKnot<T>> stages;

  size_t horizon() const noexcept { return stages.size(); }
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./lqr-knot.txx"
#endif
