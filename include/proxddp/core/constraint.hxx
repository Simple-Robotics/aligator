/// @file constraint.hxx
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/constraint.hpp"

namespace proxddp {

template <typename Scalar>
void ConstraintStackTpl<Scalar>::pushBack(const ConstraintType &el,
                                          const long nr) {
  const long last_cursor = indices_.back();
  storage_.push_back(el);
  indices_.push_back(last_cursor + nr);
  dims_.push_back(nr);
  total_dim_ += nr;
}

template <typename Scalar>
void ConstraintStackTpl<Scalar>::pushBack(const ConstraintType &el) {
  assert(el.func != 0 && "constraint must have non-null underlying function.");
  assert(el.set != 0 && "constraint must have non-null underlying set.");
  const long nr = el.func->nr;
  pushBack(el, nr);
}

} // namespace proxddp
