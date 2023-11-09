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
template <typename Scalar> struct LQRKnotTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  uint nx, nu, nc;
  MatrixXs Q, S, R;
  VectorXs q, r;
  MatrixXs A, B, E;
  VectorXs f;
  MatrixXs C, D;
  VectorXs d;

  uint nth;
  MatrixXs Gammath;
  MatrixXs Gammax;
  MatrixXs Gammau;
  VectorXs gamma;

  LQRKnotTpl(uint nx, uint nu, uint nc)
      : nx(nx), nu(nu), nc(nc),                        //
        Q(nx, nx), S(nx, nu), R(nu, nu), q(nx), r(nu), //
        A(nx, nx), B(nx, nu), E(nx, nx), f(nx),        //
        C(nc, nx), D(nc, nu), d(nc), nth(0) {
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

  inline void addParameterization(uint nth) {
    this->nth = nth;
    Gammath.setZero(nth, nth);
    Gammax.setZero(nx, nth);
    Gammau.setZero(nu, nth);
    gamma.setZero(nth);
  }
};

template <typename Scalar> struct LQRProblemTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  std::vector<LQRKnotTpl<Scalar>> stages;
  MatrixXs G0;
  VectorXs g0;

  inline int horizon() const noexcept { return int(stages.size()) - 1; }
  inline uint nc0() const noexcept { return (uint)g0.rows(); }

  LQRProblemTpl() : stages(), G0(), g0() {}

  LQRProblemTpl(const std::vector<LQRKnotTpl<Scalar>> &knots, long nc0)
      : stages(knots), G0(), g0(nc0) {
    assert(stages.size() > 0);
    auto nx0 = stages[0].nx;
    G0.resize(nc0, nx0);
  }

  void addParameterization(uint nth) {
    for (uint i = 0; i <= horizon(); i++) {
      stages[i].addParameterization(nth);
    }
  }

  inline bool isParameterized() const {
    return isInitialized() && (stages[0].nth > 0);
  }

  inline bool isInitialized() const { return !stages.empty(); }
};

/// @brief Fill in a KKT constraint matrix and vector for the given LQ problem
/// with the given dual-regularization parameters @p mudyn and @p mueq.
/// @returns Whether the matrices were successfully allocated.
template <typename Scalar>
bool lqrDenseMatrix(const LQRProblemTpl<Scalar> &problem, Scalar mudyn,
                    Scalar mueq, typename math_types<Scalar>::MatrixXs &mat,
                    typename math_types<Scalar>::VectorXs &rhs) {
  using knot_t = LQRKnotTpl<Scalar>;
  const std::vector<knot_t> &knots = problem.stages;
  size_t N = (size_t)problem.horizon();

  if (!problem.isInitialized())
    return false;

  mat.setZero();

  uint idx = 0;
  {
    uint nc0 = problem.nc0();
    uint nx0 = problem.stages[0].nx;
    mat.block(nc0, 0, nx0, nc0) = problem.G0.transpose();
    mat.block(0, nc0, nc0, nx0) = problem.G0;
    mat.topLeftCorner(nc0, nc0).diagonal().setConstant(-mudyn);

    rhs.head(nc0) = problem.g0;
    idx += nc0;
  }

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
  return true;
}

/// @copybrief lqrDenseMatrix()
template <typename Scalar>
auto lqrDenseMatrix(const LQRProblemTpl<Scalar> &problem, Scalar mudyn,
                    Scalar mueq) {

  decltype(auto) knots = problem.stages;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using knot_t = LQRKnotTpl<Scalar>;
  uint nc0 = problem.nc0();
  size_t N = knots.size() - 1UL;
  uint nrows = nc0;
  for (size_t t = 0; t <= N; t++) {
    const knot_t &model = knots[t];
    nrows += model.nx + model.nu + model.nc;
    if (t != N)
      nrows += model.nx;
  }

  MatrixXs mat(nrows, nrows);
  VectorXs rhs(nrows);

  if (!lqrDenseMatrix(problem, mudyn, mueq, mat, rhs)) {
    ALIGATOR_RUNTIME_ERROR("Problem was not initialized.");
  }
  return std::make_pair(mat, rhs);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const LQRKnotTpl<Scalar> &self) {
  oss << "LQRKnot {";
  oss << fmt::format("\n  nx:  {:d}", self.nx) //
      << fmt::format("\n  nu:  {:d}", self.nu) //
      << fmt::format("\n  nc:  {:d}", self.nc);
  if (self.nth > 0) {
    oss << fmt::format("\n  nth: {:d}", self.nth);
  }
#ifndef NDEBUG
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
  if (self.nth > 0) {
    oss << eigenPrintWithPreamble(self.Gammax, "\n  Gammax: ") //
        << eigenPrintWithPreamble(self.Gammau, "\n  Gammau: ") //
        << eigenPrintWithPreamble(self.gamma, "\n  gamma: ");
  }
#endif
  oss << "\n}";
  return oss;
}

} // namespace gar
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./lqr-problem.txx"
#endif
