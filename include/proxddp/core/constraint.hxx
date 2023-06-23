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
Eigen::VectorBlock<Derived, -1> ConstraintStackTpl<Scalar>::segmentByConstraint(
    const Eigen::MatrixBase<Derived> &lambda, const std::size_t j) const {
  Derived &lam_cast = lambda.const_cast_derived();
  assert(lambda.size() == totalDim());
  return lam_cast.segment(getIndex(j), getDim(j));
}

template <typename Scalar>
template <typename Derived>
Eigen::VectorBlock<const Derived, -1>
ConstraintStackTpl<Scalar>::constSegmentByConstraint(
    const Eigen::MatrixBase<Derived> &lambda, const std::size_t j) const {
  assert(lambda.size() == totalDim());
  return lambda.segment(getIndex(j), getDim(j));
}

template <typename Scalar>
template <typename Derived>
auto ConstraintStackTpl<Scalar>::rowsByConstraint(
    const Eigen::MatrixBase<Derived> &J_, const std::size_t j) const {
  using MatrixType = Eigen::MatrixBase<Derived>;
  MatrixType &J = const_cast<MatrixType &>(J_);
  assert(J.rows() == totalDim());
  return J.middleRows(getIndex(j), getDim(j));
}

} // namespace proxddp
