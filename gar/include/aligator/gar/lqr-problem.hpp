/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"
#include "aligator/memory/allocator.hpp"
#include "tags.hpp"
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
template <typename Scalar> struct LqrKnotTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  static constexpr int Alignment = Eigen::AlignedMax;
  using VectorMap = Eigen::Map<VectorXs, Alignment>;
  using MatrixMap = Eigen::Map<MatrixXs, Alignment>;
  using allocator_type = polymorphic_allocator;

  uint nx;
  uint nu;
  uint nc;
  uint nx2;
  uint nth;

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

  LqrKnotTpl(uint nx, uint nu, uint nc, uint nx2, uint nth,
             allocator_type alloc = {});

  /// @brief Delegating constructor, assumes @ref nth = 0.
  LqrKnotTpl(uint nx, uint nu, uint nc, uint nx2, allocator_type alloc = {})
      : LqrKnotTpl(nx, nu, nc, nx2, 0, alloc) {}

  /// @brief Delegating constructor, assumes @ref nx2 = nx, and @ref nth = 0.
  LqrKnotTpl(uint nx, uint nu, uint nc, allocator_type alloc = {})
      : LqrKnotTpl(nx, nu, nc, nx, 0, alloc) {}

  /// @brief Copy constructor. Allocator must be given.
  LqrKnotTpl(const LqrKnotTpl &other, allocator_type alloc = {});
  /// @brief Move constructor. Allocator will be moved from other. Other will be
  /// have @ref m_empty_after_move set to true.
  LqrKnotTpl(LqrKnotTpl &&other);
  /// @brief Copy assignment. Current allocator will be reused if required.
  LqrKnotTpl &operator=(const LqrKnotTpl &other);
  /// @brief Move assignment. Other allocator will be stolen.
  LqrKnotTpl &operator=(LqrKnotTpl &&);

  ~LqrKnotTpl();

  void assign(const LqrKnotTpl<Scalar> &other);

  // reallocates entire buffer for contigousness
  LqrKnotTpl &addParameterization(uint nth);

  bool isApprox(const LqrKnotTpl &other,
                Scalar prec = std::numeric_limits<Scalar>::epsilon()) const;

  friend bool operator==(const LqrKnotTpl &lhs, const LqrKnotTpl &rhs) {
    return lhs.isApprox(rhs);
  }

  allocator_type get_allocator() const { return m_allocator; }

  inline bool empty_after_move() const { return m_empty_after_move; }

private:
  explicit LqrKnotTpl(no_alloc_t, allocator_type alloc = {});
  /// Deallocation helper. Used in constructor and copy assignment op.
  void deallocate();
  /// Whether the current knot is not allocated
  bool m_empty_after_move{true};
  allocator_type m_allocator;
};

template <typename Scalar> struct LqrProblemTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  static constexpr int Alignment = Eigen::AlignedMax;
  using KnotType = LqrKnotTpl<Scalar>;
  using KnotVector = std::pmr::vector<KnotType>;
  using allocator_type = polymorphic_allocator;
  using VectorMap = Eigen::Map<VectorXs, Alignment>;
  using MatrixMap = Eigen::Map<MatrixXs, Alignment>;
  KnotVector stages;
  MatrixMap G0;
  VectorMap g0;

  inline int horizon() const noexcept { return (int)stages.size() - 1; }
  /// @brief Dimension of the initial condition constraint.
  inline uint nc0() const noexcept { return (uint)g0.rows(); }

  explicit LqrProblemTpl(allocator_type alloc = {})
      : stages(alloc), G0(NULL, 0, 0), g0(NULL, 0) {}

  /// @brief This constructor will take the knots as-is, copying their specified
  /// allocator.
  LqrProblemTpl(const KnotVector &knots, long nc0);
  LqrProblemTpl(KnotVector &&knots, long nc0);

  LqrProblemTpl(const LqrProblemTpl &other) = delete;
  /// @brief Move constructor - we steal the allocator from the source object.
  LqrProblemTpl(LqrProblemTpl &&other);

  ~LqrProblemTpl();

  void addParameterization(uint nth) {
    if (stages.empty())
      return;
    for (uint i = 0; i <= (uint)horizon(); i++) {
      stages[i].addParameterization(nth);
    }
  }

  inline bool isParameterized() const {
    return !stages.empty() && (stages[0].nth > 0);
  }

  inline bool isInitialized() const { return !stages.empty() && !m_is_invalid; }

  inline uint ntheta() const { return stages[0].nth; }

  /// Evaluate the quadratic objective.
  Scalar evaluate(const VectorOfVectors &xs, const VectorOfVectors &us,
                  const std::optional<ConstVectorRef> &theta_) const;

  allocator_type get_allocator() const { return stages.get_allocator(); }

private:
  /// internal. tag object as empty after move op (or before initialization)
  bool m_is_invalid{true};
};

template <typename Scalar>
bool lqrKnotsSameDim(const LqrKnotTpl<Scalar> &lhs,
                     const LqrKnotTpl<Scalar> &rhs) {
  return (lhs.nx == rhs.nx) && (lhs.nu == rhs.nu) && (lhs.nc == rhs.nc) &&
         (lhs.nx2 == rhs.nx2) && (lhs.nth == rhs.nth);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss, const LqrKnotTpl<Scalar> &self) {
  oss << "LqrKnot {";
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
