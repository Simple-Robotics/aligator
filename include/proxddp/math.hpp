#pragma once
/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxddp/utils/exceptions.hpp"
#include <proxnlp/math.hpp>

#include <cmath>
#include <type_traits>

#define PROXDDP_RAISE_IF_NAN(value)                                            \
  if (::proxddp::math::check_value(value))                                     \
  PROXDDP_RUNTIME_ERROR("Encountered NaN.\n")

#define PROXDDP_RAISE_IF_NAN_NAME(value, name)                                 \
  if (::proxddp::math::check_value(value))                                     \
  PROXDDP_RUNTIME_ERROR(                                                       \
      fmt::format("Encountered NaN for variable {:s}\n", name))

namespace proxddp {
/// Math utilities
namespace math {

// NOLINTBEGIN(misc-unused-using-decls)
using proxnlp::math::check_scalar;
using proxnlp::math::infty_norm;
using proxnlp::math::scalar_close;
// NOLINTEND(misc-unused-using-decls)

/// @brief  Check if a numerical value or vector contains NaNs or infinite
/// elements. Returns true if so.
template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
bool check_value(const T &s) {
  return check_scalar(s);
}

/// @copybrief check_value()
template <typename MatrixType>
bool check_value(const Eigen::MatrixBase<MatrixType> &x) {
  return (x.hasNaN() || (!x.allFinite()));
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

template <typename T> void setZero(std::vector<T> &mats) {
  for (std::size_t i = 0; i < mats.size(); i++) {
    mats[i].setZero();
  }
}

/// @brief Computes the inertia of a diagonal matrix \f$D\f$ represented by its
/// diagonal vector.
/// @param[out] output Triplet (n+, n0, n-) of number of positive, zero or
/// negative eigenvalues.
template <typename VectorType>
void compute_inertia(const VectorType &v, std::size_t *inertia) {
  static_assert(VectorType::ColsAtCompileTime == 1,
                "VectorType should be a vector.");
  std::size_t &numpos = inertia[0];
  std::size_t &numzer = inertia[1];
  std::size_t &numneg = inertia[2];
  numpos = 0;
  numzer = 0;
  numneg = 0;
  const Eigen::Index n = v.size();
  auto s = v.cwiseSign().template cast<int>();

  for (Eigen::Index i = 0; i < n; i++) {
    switch (s(i)) {
    case 1:
      numpos++;
      break;
    case -1:
      numneg++;
      break;
    case 0:
      numzer++;
      break;
    default:
      PROXDDP_RUNTIME_ERROR("Vector sign() should be -1, 0, or 1.");
    }
  }
}

} // namespace math
} // namespace proxddp
