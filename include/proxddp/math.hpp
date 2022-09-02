#pragma once
/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxddp/utils/exceptions.hpp"
#include <proxnlp/math.hpp>

#include <cmath>
#include <type_traits>

#define proxddp_raise_if_nan(value)                                            \
  if (::proxddp::math::checkScalar(value))                                     \
  proxddp_runtime_error("encountered NaN.\n")

namespace proxddp {
/// Math utilities
namespace math {

using namespace proxnlp::math;

/// @brief Computes the inertia of a diagonal matrix \f$D\f$ represented by its
/// diagonal vector.
/// @param[out] output Triplet (n+, n0, n-) of number of positive, zero or
/// negative eigenvalues.
template <typename VectorType>
void compute_inertia(const VectorType &v, unsigned int *output) {
  static_assert(VectorType::ColsAtCompileTime == 1);
  unsigned int &numpos = output[0];
  unsigned int &numzer = output[1];
  unsigned int &numneg = output[2];
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
      proxddp_runtime_error("Vector sign() should be -1, 0, or 1.");
    }
  }
}

} // namespace math
} // namespace proxddp
