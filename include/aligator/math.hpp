/// @file math.hpp
/// @brief Math utilities.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/utils/exceptions.hpp"
#include <proxsuite-nlp/math.hpp>

#define ALIGATOR_RAISE_IF_NAN(value)                                           \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN.\n")

#define ALIGATOR_RAISE_IF_NAN_NAME(value, name)                                \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR(                                                      \
      fmt::format("Encountered NaN for variable {:s}\n", name))

#define ALIGATOR_DYNAMIC_TYPEDEFS(Scalar) PROXSUITE_NLP_DYNAMIC_TYPEDEFS(Scalar)

namespace aligator {

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

} // namespace math
} // namespace aligator
