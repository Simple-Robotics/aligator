/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/math.hpp"
#include "aligator/gar/fwd.hpp"
#include "aligator/memory/arena-matrix.hpp"
#include "aligator/tags.hpp"
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
  using MVec = ArenaMatrix<VectorXs>;
  using MMat = ArenaMatrix<MatrixXs>;
  using allocator_type = polymorphic_allocator;

  uint nx;
  uint nu;
  uint nc;
  uint nx2;
  uint nth;

  MMat Q, S, R;
  MVec q, r;
  MMat A, B, E;
  MVec f;
  MMat C, D;
  MVec d;

  MMat Gth;
  MMat Gx;
  MMat Gu;
  MMat Gv;
  MVec gamma;

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
  LqrKnotTpl(LqrKnotTpl &&other) noexcept;
  /// @brief Extended move constructor.
  LqrKnotTpl(LqrKnotTpl &&other, const allocator_type &alloc);
  /// @brief Copy assignment. Current allocator will be reused if required.
  LqrKnotTpl &operator=(const LqrKnotTpl &other);
  /// @brief Move assignment. Other allocator will be stolen.
  LqrKnotTpl &operator=(LqrKnotTpl &&);

  ~LqrKnotTpl() = default;

  /// \brief Assign matrices (and dimensions) from another LqrKnotTpl.
  void assign(const LqrKnotTpl<Scalar> &other);

  // reallocates entire buffer for contigousness
  LqrKnotTpl &addParameterization(uint nth);

  bool isApprox(const LqrKnotTpl &other,
                Scalar prec = std::numeric_limits<Scalar>::epsilon()) const;

  friend bool operator==(const LqrKnotTpl &lhs, const LqrKnotTpl &rhs) {
    return lhs.isApprox(rhs);
  }

  allocator_type get_allocator() const { return m_allocator; }

private:
  explicit LqrKnotTpl(no_alloc_t, allocator_type alloc = {});
  allocator_type m_allocator;
};

template <typename Scalar> struct LqrProblemTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  static constexpr int Alignment = Eigen::AlignedMax;
  using KnotType = LqrKnotTpl<Scalar>;
  using KnotVector = std::pmr::vector<KnotType>;
  using allocator_type = polymorphic_allocator;
  using MVec = ArenaMatrix<VectorXs>;
  using MMat = ArenaMatrix<MatrixXs>;
  MMat G0;
  MVec g0;
  KnotVector stages;

  inline int horizon() const noexcept { return (int)stages.size() - 1; }
  /// @brief Dimension of the initial condition constraint.
  inline uint nc0() const noexcept { return (uint)g0.rows(); }

  explicit LqrProblemTpl(allocator_type alloc = {})
      : G0(alloc)
      , g0(alloc)
      , stages(alloc) {
    assert(check_allocators());
  }

  /// @brief This constructor will take the knots as-is.
  LqrProblemTpl(const KnotVector &knots, long nc0, allocator_type alloc = {});
  /// @brief This constructor will take the knots as-is, copying their specified
  /// allocator.
  LqrProblemTpl(KnotVector &&knots, long nc0);

  /// @brief Copy constructor. Will copy the allocator from @p other.
  LqrProblemTpl(const LqrProblemTpl &other, allocator_type alloc = {})
      : LqrProblemTpl(other.stages, other.nc0(), alloc) {
    this->G0 = other.G0;
    this->g0 = other.g0;
  }

  /// @brief Move constructor - we steal the allocator from the source object.
  LqrProblemTpl(LqrProblemTpl &&other)
      : LqrProblemTpl(std::move(other.stages), other.nc0()) {
    this->G0 = other.G0;
    this->g0 = other.g0;
  }

  LqrProblemTpl &operator=(LqrProblemTpl &&other) {
    this->G0 = std::move(other.G0);
    this->g0 = std::move(other.g0);
    this->stages = std::move(other.stages);
    return *this;
  }

  ~LqrProblemTpl() = default;

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

  inline bool isInitialized() const { return !stages.empty(); }

  inline uint ntheta() const { return stages[0].nth; }

  inline bool isApprox(const LqrProblemTpl &other) {
    if (horizon() != other.horizon() || !G0.isApprox(other.G0) ||
        !g0.isApprox(other.g0))
      return false;
    for (uint i = 0; i < uint(horizon()); i++) {
      if (!stages[i].isApprox(other.stages[i]))
        return false;
    }
    return true;
  }

  /// Evaluate the quadratic objective.
  Scalar evaluate(const VectorOfVectors &xs, const VectorOfVectors &us,
                  const std::optional<ConstVectorRef> &theta_) const;

  allocator_type get_allocator() const { return G0.get_allocator(); }

private:
  /// Check consistency of all allocators.
  [[nodiscard]] bool check_allocators() const {
    return get_allocator() == g0.get_allocator() &&
           get_allocator() == stages.get_allocator();
  }
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

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct LqrKnotTpl<context::Scalar>;
extern template struct LqrProblemTpl<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
