/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"
#include "mem-req.hpp"
#include <fmt/format.h>

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
  enum { Alignment = Eigen::AlignedMax };
  using VectorMap = Eigen::Map<VectorXs, Alignment>;
  using MatrixMap = Eigen::Map<MatrixXs, Alignment>;

  // tag type
  struct no_alloc_t {
    explicit constexpr no_alloc_t() {}
  };
  static constexpr no_alloc_t no_alloc{};

  uint nx, nu, nc, nx2, nth;

  MatrixMap Q, S, R;
  VectorMap q, r;
  MatrixMap A, B, E;
  VectorMap f;
  MatrixMap C, D;
  VectorMap d;

  MatrixMap Gth;
  MatrixMap Gx;
  MatrixMap Gu;
  MatrixMap Gv;
  VectorMap gamma;

  LQRKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth = 0);

  LQRKnotTpl(uint nx, uint nu, uint nc) : LQRKnotTpl(nx, nu, nc, nx) {}

  void allocate();
  // initialize the matrices.
  void initialize();
  // reallocates entire buffer for contigousness
  void addParameterization(uint nth);

  LQRKnotTpl(const LQRKnotTpl &other);
  LQRKnotTpl(LQRKnotTpl &&other);
  LQRKnotTpl &operator=(const LQRKnotTpl &other);
  LQRKnotTpl &operator=(LQRKnotTpl &&other);

  ~LQRKnotTpl();

  friend void swap(LQRKnotTpl &lhs, LQRKnotTpl &rhs) {
    using std::swap;
    swap(lhs.nx, rhs.nx);
    swap(lhs.nu, rhs.nu);
    swap(lhs.nc, rhs.nc);
    swap(lhs.nx2, rhs.nx2);
    swap(lhs.nth, rhs.nth);
    // only swap the memory ptr, do not swap the Eigen::Map objects.
    swap(lhs.memory, rhs.memory);
    swap(lhs.req, rhs.req);

    lhs.initialize();
    rhs.initialize();
  }

  bool isApprox(const LQRKnotTpl &other,
                Scalar prec = std::numeric_limits<Scalar>::epsilon()) const;

  friend bool operator==(const LQRKnotTpl &lhs, const LQRKnotTpl &rhs) {
    return lhs.isApprox(rhs);
  }

private:
  LQRKnotTpl(no_alloc_t, uint nx, uint nu, uint nc, uint nx2, uint nth);
  Scalar *memory;
  MemReq req;
};

template <typename Scalar> LQRKnotTpl<Scalar>::~LQRKnotTpl() {
  if (memory)
    std::free(memory);
}

template <typename Scalar> struct LQRProblemTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using KnotType = LQRKnotTpl<Scalar>;
  using KnotVector = std::vector<KnotType>;
  KnotVector stages;
  MatrixXs G0;
  VectorXs g0;

  inline int horizon() const noexcept { return int(stages.size()) - 1; }
  inline uint nc0() const noexcept { return (uint)g0.rows(); }

  LQRProblemTpl() : stages(), G0(), g0() {}

  LQRProblemTpl(KnotVector &&knots, long nc0) : stages(knots), G0(), g0(nc0) {
    initialize();
  }

  LQRProblemTpl(const KnotVector &knots, long nc0)
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
bool lqrKnotsSameDim(const LQRKnotTpl<Scalar> &lhs,
                     const LQRKnotTpl<Scalar> &rhs) {
  return (lhs.nx == rhs.nx) && (lhs.nu == rhs.nu) && (lhs.nc == rhs.nc) &&
         (lhs.nx2 == rhs.nx2) && (lhs.nth == rhs.nth);
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
#include "lqr-problem.txx"
#endif
