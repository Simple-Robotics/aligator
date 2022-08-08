#pragma once
/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <proxnlp/math.hpp>

#include <cmath>

namespace proxddp {
/// Math utilities
namespace math {

using namespace proxnlp::math;

/// @brief Check that a scalar is neither inf, nor NaN.
template <typename Scalar> inline bool checkScalar(const Scalar value) {
  return std::isnan(value) || std::isinf(value);
}

/**
 * @brief Tests whether @p a and @p b are close, within absolute and relative
 * precision @p prec.
 */
template <typename Scalar>
bool scalar_close(const Scalar a, const Scalar b, const Scalar prec) {
  return std::abs(a - b) < prec * (1 + std::max(std::abs(a), std::abs(b)));
}

} // namespace math
} // namespace proxddp
