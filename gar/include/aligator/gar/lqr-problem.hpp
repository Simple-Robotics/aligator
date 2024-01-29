/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"

#include <optional>

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

  uint nx, nu, nc, nx2;
  MatrixXs Q, S, R;
  VectorXs q, r;
  MatrixXs A, B, E;
  VectorXs f;
  MatrixXs C, D;
  VectorXs d;

  uint nth;
  MatrixXs Gth;
  MatrixXs Gx;
  MatrixXs Gu;
  MatrixXs Gv;
  VectorXs gamma;

  LQRKnotTpl(uint nx, uint nu, uint nc, uint nx2)
      : nx(nx), nu(nu), nc(nc), nx2(nx2),              //
        Q(nx, nx), S(nx, nu), R(nu, nu), q(nx), r(nu), //
        A(nx2, nx), B(nx2, nu), E(nx2, nx), f(nx2),    //
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

  LQRKnotTpl(uint nx, uint nu, uint nc) : LQRKnotTpl(nx, nu, nc, nx) {}

  inline void addParameterization(uint nth) {
    this->nth = nth;
    Gth.setZero(nth, nth);
    Gx.setZero(nx, nth);
    Gu.setZero(nu, nth);
    Gv.setZero(nc, nth);
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

  LQRProblemTpl(std::vector<LQRKnotTpl<Scalar>> &&knots, long nc0)
      : stages(knots), G0(), g0(nc0) {
    initialize();
  }

  LQRProblemTpl(const std::vector<LQRKnotTpl<Scalar>> &knots, long nc0)
      : stages(knots), G0(), g0(nc0) {
    initialize();
  }

  void addParameterization(uint nth) {
    if (!isInitialized())
      return;
    for (uint i = 0; i <= (uint)horizon(); i++) {
      stages[i].addParameterization(nth);
    }
  }

  inline bool isParameterized() const {
    return isInitialized() && (stages[0].nth > 0);
  }

  inline bool isInitialized() const { return !stages.empty(); }

  inline uint ntheta() const { return stages[0].nth; }

  /// Evaluate the quadratic objective.
  Scalar evaluate(const VectorOfVectors &xs, const VectorOfVectors &us,
                  const std::optional<ConstVectorRef> &theta_) const;

protected:
  void initialize() {
    assert(isInitialized());
    auto nx0 = stages[0].nx;
    G0.resize(nc0(), nx0);
  }
};

template <typename Scalar>
Scalar LQRProblemTpl<Scalar>::evaluate(
    const VectorOfVectors &xs, const VectorOfVectors &us,
    const std::optional<ConstVectorRef> &theta_) const {
  if ((int)xs.size() != horizon() + 1)
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Wrong size for vector xs (expected {:d}).", horizon() + 1));
  if ((int)us.size() < horizon())
    ALIGATOR_RUNTIME_ERROR(
        fmt::format("Wrong size for vector us (expected {:d}).", horizon()));

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
    oss << eigenPrintWithPreamble(self.Gth, "\n  Gth: ") //
        << eigenPrintWithPreamble(self.Gx, "\n  Gx: ")   //
        << eigenPrintWithPreamble(self.Gu, "\n  Gu: ")   //
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
