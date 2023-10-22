/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"

namespace aligator {
namespace gar {

/// @brief Struct describing a stage of a constrained LQ problem.
///
/// A LQ knot corresponding to cost
/// \f\[
///   \frac{1}{2}
///   \begin{bmatrix}x \\ u\end{bmatrix}^\top
///   \begin{bmatrix}Q & S \\ S^\top & R\end{bmatrix}
///   \begin{bmatrix}x \\ u\end{bmatrix}
///   + q^\top x + r^\top u
/// \f\]
/// and constraints
/// \f\[
///   Ex' + Ax + Bu + f = 0, \quad
///   Cx + Du + d = 0.
/// \f\]
///
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

template <typename Scalar> struct LQRProblem {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using knot_t = LQRKnot<Scalar>;
  std::vector<knot_t> stages;
  MatrixXs G0;
  VectorXs g0;

  long horizon() const noexcept { return long(stages.size()) - 1L; }

  LQRProblem(const std::vector<knot_t> &knots, long nc0)
      : stages(knots), G0(), g0(nc0) {
    assert(stages.size() > 0);
    auto nx0 = stages[0].nx;
    G0.resize(nc0, nx0);
  }
};

template <typename Scalar>
void lqrDenseMatrix(const std::vector<LQRKnot<Scalar>> &knots, Scalar mudyn,
                    Scalar mueq, typename math_types<Scalar>::MatrixXs &mat,
                    typename math_types<Scalar>::VectorXs &rhs) {
  using knot_t = LQRKnot<Scalar>;
  size_t N = knots.size() - 1UL;

  uint idx = 0;
  for (size_t t = 0; t <= N; t++) {
    const knot_t &model = knots[t];
    // get block for current variables
    uint n = model.nx + model.nu + model.nc;
    auto block = mat.block(idx, idx, n, n);
    auto rhsblk = rhs.segment(idx, n);
    auto Q = block.topLeftCorner(model.nx, model.nx);
    auto St = block.leftCols(model.nx).middleRows(model.nx, model.nu);
    auto R = block.block(model.nx, model.nx, model.nu, model.nu);
    auto C = block.bottomRows(model.nc).leftCols(model.nx);
    auto D = block.bottomRows(model.nc).middleCols(model.nx, model.nu);
    auto dual = block.bottomRightCorner(model.nc, model.nc).diagonal();
    dual.array() = -mueq;

    Q = model.Q;
    St = model.S.transpose();
    R = model.R;
    C = model.C;
    D = model.D;

    block = block.template selfadjointView<Eigen::Lower>();

    rhsblk.head(model.nx) = model.q;
    rhsblk.segment(model.nx, model.nu) = model.r;
    rhsblk.tail(model.nc) = model.d;

    // fill in dynamics
    // row contains [A; B; 0; -mu*I, E] -> nx + nu + nc + 2*nx cols
    if (t != N) {
      auto row = mat.block(idx + n, idx, model.nx, model.nx * 2 + n);
      row.leftCols(model.nx) = model.A;
      row.middleCols(model.nx, model.nu) = model.B;
      row.middleCols(n, model.nx).diagonal().array() = -mudyn;
      row.rightCols(model.nx) = model.E;

      rhs.segment(idx + n, model.nx) = model.f;

      auto col =
          mat.transpose().block(idx + n, idx, model.nx, model.nx * 2 + n);
      col = row;

      // shift by size of block + multiplier size
      idx += model.nx + n;
    }
  }
}

template <typename Scalar>
auto lqrDenseMatrix(const std::vector<LQRKnot<Scalar>> &knots, Scalar mudyn,
                    Scalar mueq) {

  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using knot_t = LQRKnot<Scalar>;
  size_t N = knots.size() - 1UL;
  uint nrows = 0;
  for (size_t t = 0; t <= N; t++) {
    const knot_t &model = knots[t];
    nrows += model.nx + model.nu + model.nc;
    if (t != N)
      nrows += model.nx;
  }

  MatrixXs mat(nrows, nrows);
  mat.setZero();
  VectorXs rhs(nrows);

  lqrDenseMatrix(knots, mudyn, mueq, mat, rhs);
  return std::make_pair(mat, rhs);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const LQRKnot<Scalar> &self) {
  oss << "LQRKnot {";
#ifdef NDEBUG
  oss << fmt::format("\n  nx: {:d}", self.nx) //
      << fmt::format("\n  nu: {:d}", self.nu) //
      << fmt::format("\n  nc: {:d}", self.nc);
#else
  oss << eigenPrintWithPreamble(self.Q, "\n  Q: ") //
      << eigenPrintWithPreamble(self.S, "\n  S: ") //
      << eigenPrintWithPreamble(self.R, "\n  R: ") //
      << eigenPrintWithPreamble(self.q, "\n  q: ") //
      << eigenPrintWithPreamble(self.r, "\n  r: ");

  oss << eigenPrintWithPreamble(self.A, "\n  A: ") //
      << eigenPrintWithPreamble(self.B, "\n  B: ") //
      << eigenPrintWithPreamble(self.E, "\n  E: ") //
      << eigenPrintWithPreamble(self.f, "\n  f: ");

  oss << eigenPrintWithPreamble(self.C, "\n  C: ") //
      << eigenPrintWithPreamble(self.D, "\n  D: ") //
      << eigenPrintWithPreamble(self.d, "\n  d: ");
#endif
  oss << "\n}";
  return oss;
}

} // namespace gar
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./lqr-knot.txx"
#endif
