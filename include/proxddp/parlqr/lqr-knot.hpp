#pragma once

#include "proxddp/math.hpp"

namespace proxddp {

template <typename Scalar> struct LQRKnot {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  uint nx, nu, nc;
  MatrixXs Q, S, R;
  VectorXs q, r;
  MatrixXs A, B, E;
  VectorXs f;
  MatrixXs C, D;
  VectorXs d;

  LQRKnot(uint nx, uint nu, uint nc)
      : nx(nx), nu(nu), nc(nc),
        //
        Q(nx, nx), S(nx, nu), R(nu, nu), q(nx), r(nu),
        //
        A(nx, nx), B(nx, nu), E(nx, nx), f(nx),
        //
        C(nc, nx), D(nc, nu), d(nc) {}

  uint ntot() const { return nx + nu + nc; }
};

} // namespace proxddp
