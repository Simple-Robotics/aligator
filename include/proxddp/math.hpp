#pragma once
/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <cmath>

namespace proxddp {
/// Math utilities
namespace math {

using namespace proxnlp::math;

/// @brief Check that a scalar is neither inf, nor NaN.
template <typename Scalar> inline bool checkScalar(const Scalar value) {
  return std::isnan(value) || std::isinf(value);
}

} // namespace math
} // namespace proxddp
