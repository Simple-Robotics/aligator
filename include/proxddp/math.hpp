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

// NOLINTBEGIN(misc-unused-using-decls)
using proxnlp::math_types;
// NOLINTEND(misc-unused-using-decls)

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

} // namespace math
} // namespace proxddp
