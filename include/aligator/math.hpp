/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include <Eigen/Core>

#define ALIGATOR_DYNAMIC_TYPEDEFS(Scalar)                                      \
  using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;                   \
  using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;      \
  using VectorOfVectors = std::vector<VectorXs>;                               \
  using VectorRef = Eigen::Ref<VectorXs>;                                      \
  using MatrixRef = Eigen::Ref<MatrixXs>;                                      \
  using ConstVectorRef = Eigen::Ref<const VectorXs>;                           \
  using ConstMatrixRef = Eigen::Ref<const MatrixXs>;                           \
  using Vector3s = Eigen::Matrix<Scalar, 3, 1>;                                \
  using Vector6s = Eigen::Matrix<Scalar, 6, 1>;                                \
  using Matrix3Xs = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;                  \
  using Matrix6Xs = Eigen::Matrix<Scalar, 6, Eigen::Dynamic>;                  \
  using Matrix6s = Eigen::Matrix<Scalar, 6, 6>

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

/// This type class recognises whether
/// @tparam Base Base template (CRTP) class
/// @tparam Derived Derived class, does not need to derive from `Base<Derived>`
/// for this type trait to evaluate to true.
template <template <class> class Base, typename Derived> struct is_tpl_base_of {
  static constexpr std::false_type f(const void *);
  template <typename OtherDerived>
  static constexpr std::true_type f(const Base<OtherDerived> *);
  static constexpr bool value =
      decltype(f(std::declval<std::remove_reference_t<Derived> *>()))::value;
};

template <template <class> class Base, typename Derived>
inline constexpr bool is_tpl_base_of_v = is_tpl_base_of<Base, Derived>::value;

template <typename T>
struct is_eigen : std::bool_constant<is_tpl_base_of_v<Eigen::EigenBase, T>> {};

template <typename T> inline constexpr bool is_eigen_v = is_eigen<T>::value;

template <typename T>
using enable_if_eigen_t = std::enable_if_t<is_eigen_v<std::decay_t<T>>>;

template <typename T>
inline constexpr bool is_eigen_dense_v = is_tpl_base_of_v<Eigen::DenseBase, T>;

template <typename T, typename T2 = void>
using enable_if_eigen_dense_t = std::enable_if_t<is_eigen_dense_v<T>, T2>;

template <typename T>
inline constexpr bool is_eigen_matrix_v =
    is_tpl_base_of_v<Eigen::MatrixBase, T>;

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

///  @brief  Typedefs for math (Eigen vectors, matrices) depending on scalar
/// type.
template <typename _Scalar> struct math_types {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(_Scalar);
};

/// Prints an Eigen object using Eigen::IOFormat
/// with a piece of text prepended and all rows shifted
/// by the length of that text.
template <typename D>
auto eigenPrintWithPreamble(const Eigen::EigenBase<D> &mat,
                            std::string_view text,
                            Eigen::IOFormat ft = EIGEN_DEFAULT_IO_FORMAT) {
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

template <typename MatType>
typename MatType::Scalar infty_norm(const Eigen::MatrixBase<MatType> &z) {
  if (z.rows() == 0 || z.cols() == 0) {
    return 0.;
  } else {
    return z.template lpNorm<Eigen::Infinity>();
  }
}

template <typename MatType>
typename MatType::Scalar infty_norm(const std::vector<MatType> &z) {
  const std::size_t n = z.size();
  typename MatType::Scalar out = 0.;
  for (std::size_t i = 0; i < n; i++) {
    out = std::max(out, infty_norm(z[i]));
  }
  return out;
}

/// @brief Check that a scalar is neither inf, nor NaN.
template <typename Scalar> inline bool check_scalar(const Scalar value) {
  return std::isnan(value) || std::isinf(value);
}

/// @brief    Check if a std::vector of numerical objects has invalid values.
template <typename T> bool check_value(const std::vector<T> &xs) {
  const std::size_t n = xs.size();
  for (std::size_t i = 0; i < n; i++) {
    if (check_value<T>(xs[i]))
      return true;
  }
  return false;
}

template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
bool check_value(const T &x) {
  static_assert(std::is_scalar<T>::value, "Parameter T should be scalar.");
  return check_scalar(x);
}

template <typename MatrixType>
bool check_value(const Eigen::MatrixBase<MatrixType> &x) {
  return (x.hasNaN() || (!x.allFinite()));
}

/// @brief Tests whether @p a and @p b are close, within absolute and relative
/// precision @p prec.
///
template <typename Scalar>
bool scalar_close(const Scalar a, const Scalar b,
                  const Scalar prec = std::numeric_limits<Scalar>::epsilon()) {
  return std::abs(a - b) < prec * (1 + std::max(std::abs(a), std::abs(b)));
}

template <typename T> T sign(const T &x) {
  static_assert(std::is_scalar<T>::value, "Parameter T should be scalar.");
  return T((x > T(0)) - (x < T(0)));
}

/// @brief Symmetrize a matrix using its lower triangular part.
template <typename Derived, unsigned int UpLo = Eigen::Lower>
void make_symmetric(const Eigen::MatrixBase<Derived> &matrix) {
  Derived &mat = matrix.const_cast_derived();
  // symmetrize upper part
  Eigen::SelfAdjointView<Derived, UpLo> view{mat};
  mat = view;
}

template <typename T> void setZero(std::vector<T> &mats) {
  for (std::size_t i = 0; i < mats.size(); i++) {
    mats[i].setZero();
  }
}

/// @brief Compute zi = xi + alpha * yi for all i
template <typename TA, typename AA, typename TB, typename BA, typename OutType,
          typename AOut, typename Scalar>
void vectorMultiplyAdd(const std::vector<TA, AA> &a,
                       const std::vector<TB, BA> &b,
                       std::vector<OutType, AOut> &c, const Scalar alpha) {
  assert(a.size() == b.size());
  assert(a.size() == c.size());
  const std::size_t N = a.size();
  for (std::size_t i = 0; i < N; i++) {
    c[i] = a[i] + alpha * b[i];
  }
}

} // namespace math
} // namespace aligator
