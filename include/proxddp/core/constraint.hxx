/// @file constraint.hxx
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/constraint.hpp"

namespace proxddp {

template <typename Scalar> void ConstraintStackTpl<Scalar>::clear() {
  storage_.clear();
  indices_ = {0};
  dims_.clear();
  total_dim_ = 0;
}

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
  pushBack(el, el.nr());
}

template <typename Scalar>
template <typename Derived>
auto ConstraintStackTpl<Scalar>::getRowsByConstraint(
    const Eigen::MatrixBase<Derived> &J_, const std::size_t j) const {
  using MatrixType = Eigen::MatrixBase<Derived>;
  MatrixType &J = const_cast<MatrixType &>(J_);
  assert(J.rows() == totalDim());
  return J.middleRows(getIndex(j), getDim(j));
}

} // namespace proxddp
