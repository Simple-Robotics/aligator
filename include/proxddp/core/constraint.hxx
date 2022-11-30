/// @file constraint.hpp
/// @brief Defines the constraint object for this library.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

namespace proxddp {

template <typename Scalar>
void ConstraintStackTpl<Scalar>::push_back(const ConstraintType &el,
                                           const int nr) {
  const int last_cursor = cursors_.back();
  storage_.push_back(el);
  cursors_.push_back(last_cursor + nr);
  dims_.push_back(nr);
  total_dim += nr;
}

} // namespace proxddp
