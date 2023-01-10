/// @file constraint.hxx
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

namespace proxddp {

template <typename Scalar>
void ConstraintStackTpl<Scalar>::pushBack(const ConstraintType &el,
                                          const long nr) {
  const long last_cursor = indices_.back();
  storage_.push_back(el);
  indices_.push_back(last_cursor + nr);
  dims_.push_back(nr);
  total_dim += nr;
}

} // namespace proxddp
