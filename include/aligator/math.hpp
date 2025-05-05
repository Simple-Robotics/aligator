/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include <proxsuite-nlp/math.hpp>

#define ALIGATOR_DYNAMIC_TYPEDEFS(Scalar) PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar)

#define ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar)                       \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using RowMatrixXs = typename Eigen::Transpose<MatrixXs>::PlainObject;        \
  using RowMatrixRef = Eigen::Ref<RowMatrixXs>;                                \
  using ConstRowMatrixRef = Eigen::Ref<const RowMatrixXs>

#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
/// @warning Unless using versions of Eigen past 3.4.x, NOT SUPPORTED IN
/// MULTITHREADED CONTEXTS
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)                                   \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
/// @brief Set nomalloc for the scope. Previous malloc status will be restored
/// upon exiting the scope.
/// @warning Unless using versions of Eigen past 3.4.x, NOT SUPPORTED IN
/// MULTITHREADED CONTEXTS
#define ALIGATOR_NOMALLOC_SCOPED                                               \
  const ::aligator::internal::scoped_nomalloc ___aligator_nomalloc_zone {}
/// @brief Restore the previous nomalloc status.
#define ALIGATOR_NOMALLOC_RESTORE                                              \
  ALIGATOR_EIGEN_ALLOW_MALLOC(::aligator::internal::get_cached_malloc_status())
#else
#define ALIGATOR_EIGEN_ALLOW_MALLOC(allowed)
#define ALIGATOR_NOMALLOC_SCOPED
#define ALIGATOR_NOMALLOC_RESTORE
#endif

/// @brief Entering performance-critical code.
#define ALIGATOR_NOMALLOC_BEGIN ALIGATOR_EIGEN_ALLOW_MALLOC(false)
/// @brief Exiting performance-critical code.
#define ALIGATOR_NOMALLOC_END ALIGATOR_EIGEN_ALLOW_MALLOC(true)

namespace aligator {

template <typename T>
inline constexpr bool is_eigen_dense_type =
    std::is_base_of_v<Eigen::DenseBase<T>, T>;

template <typename T>
inline constexpr bool is_eigen_matrix_type =
    std::is_base_of_v<Eigen::MatrixBase<T>, T>;

template <typename T, typename T2 = void>
using enable_if_eigen_dense = std::enable_if_t<is_eigen_dense_type<T>, T2>;

#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
namespace internal {
thread_local static bool g_cached_malloc_status = true;

inline void set_malloc_status(bool status) { g_cached_malloc_status = status; }

inline void save_malloc_status() {
  set_malloc_status(
#ifdef ALIGATOR_EIGEN_CHECK_MALLOC
      ::Eigen::internal::is_malloc_allowed()
#else
      false
#endif
  );
}

inline bool get_cached_malloc_status() { return g_cached_malloc_status; }

struct scoped_nomalloc {
  scoped_nomalloc(const scoped_nomalloc &) = delete;
  scoped_nomalloc(scoped_nomalloc &&) = delete;
  ALIGATOR_INLINE scoped_nomalloc() {
    save_malloc_status();
    ALIGATOR_EIGEN_ALLOW_MALLOC(false);
  }
  // reset to value from before the scope
  ~scoped_nomalloc() { ALIGATOR_NOMALLOC_RESTORE; }
};

} // namespace internal
#endif

// NOLINTBEGIN(misc-unused-using-decls)
using proxsuite::nlp::math_types;
// NOLINTEND(misc-unused-using-decls)

/// Prints an Eigen object using Eigen::IOFormat
/// with a piece of text prepended and all rows shifted
/// by the length of that text.
template <typename D>
auto eigenPrintWithPreamble(const Eigen::EigenBase<D> &mat,
                            const std::string &text) {
  Eigen::IOFormat ft = EIGEN_DEFAULT_IO_FORMAT;
  ft.matPrefix = text;
  ft.rowSpacer = "";
  int i = int(text.length()) - 1;
  while (i >= 0) {
    if (text[size_t(i)] != '\n')
      ft.rowSpacer += ' ';
    i--;
  }
  return mat.derived().format(ft);
}

/// Math utilities
namespace math {

// NOLINTBEGIN(misc-unused-using-decls)
using proxsuite::nlp::math::check_scalar;
using proxsuite::nlp::math::check_value;
using proxsuite::nlp::math::infty_norm;
using proxsuite::nlp::math::scalar_close;
// NOLINTEND(misc-unused-using-decls)

/// @brief    Check if a std::vector of numerical objects has invalid values.
template <typename T> bool check_value(const std::vector<T> &xs) {
  const std::size_t n = xs.size();
  for (std::size_t i = 0; i < n; i++) {
    if (check_value<T>(xs[i]))
      return true;
  }
  return false;
}

template <typename T> void setZero(std::vector<T> &mats) {
  for (std::size_t i = 0; i < mats.size(); i++) {
    mats[i].setZero();
  }
}

/// @brief Compute zi = xi + alpha * yi for all i
template <typename A, typename B, typename OutType, typename Scalar>
void vectorMultiplyAdd(const std::vector<A> &a, const std::vector<B> &b,
                       std::vector<OutType> &c, const Scalar alpha) {
  assert(a.size() == b.size());
  assert(a.size() == c.size());
  const std::size_t N = a.size();
  for (std::size_t i = 0; i < N; i++) {
    c[i] = a[i] + alpha * b[i];
  }
}

} // namespace math
} // namespace aligator
