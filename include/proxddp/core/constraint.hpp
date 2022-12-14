/// @file constraint.hpp
/// @brief Defines the constraint object for this library.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {

/// @brief Simple struct holding together a function and set, to describe a
/// constraint.
template <typename Scalar> struct StageConstraintTpl {
  shared_ptr<StageFunctionTpl<Scalar>> func;
  shared_ptr<ConstraintSetBase<Scalar>> set;
};

/// @brief Convenience class to manage a stack of constraints.
template <typename Scalar> struct ConstraintStackTpl {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintType = StageConstraintTpl<Scalar>;
  ConstraintStackTpl() : cursors_({0}){};

  std::size_t numConstraints() const { return storage_.size(); }

  inline void push_back(const ConstraintType &el) {
    assert(el.func != 0 &&
           "constraint must have non-null underlying function.");
    assert(el.set != 0 && "constraint must have non-null underlying set.");
    const int nr = el.func->nr;
    push_back(el, nr);
  }

  void push_back(const ConstraintType &el, const int nr);

  int getIndex(const std::size_t j) const { return cursors_[j]; }

  int getDim(const std::size_t j) const { return dims_[j]; }

  const ConstraintSetBase<Scalar> &getConstraintSet(const std::size_t j) const {
    return *this->storage_[j].set;
  }

  /// Get corresponding segment of a vector corresponding
  /// to the @p i-th constraint.
  template <typename Derived>
  Eigen::VectorBlock<Derived, -1>
  getSegmentByConstraint(const Eigen::MatrixBase<Derived> &lambda,
                         const std::size_t j) const {
    using MatrixType = Eigen::MatrixBase<Derived>;
    MatrixType &lam_cast = const_cast<MatrixType &>(lambda);
    assert(lambda.size() == totalDim());
    return lam_cast.segment(getIndex(j), getDim(j));
  }

  template <typename Derived>
  Eigen::VectorBlock<const Derived, -1>
  getConstSegmentByConstraint(const Eigen::MatrixBase<Derived> &lambda,
                              const std::size_t j) const {
    assert(lambda.size() == totalDim());
    return lambda.segment(getIndex(j), getDim(j));
  }

  /// Get a row-wise block of a matrix by constraint index.
  template <typename Derived>
  Eigen::Block<Derived, -1, -1>
  getBlockByConstraint(const Eigen::MatrixBase<Derived> &J_,
                       const std::size_t j) const {
    using MatrixType = Eigen::MatrixBase<Derived>;
    MatrixType &J = const_cast<MatrixType &>(J_);
    assert(J.rows() == totalDim());
    return J.middleRows(getIndex(j), getDim(j));
  }

  int totalDim() const { return total_dim; }

  /// Get the i-th constraint.
  ConstraintType &operator[](std::size_t j) {
    assert((j < this->storage_.size()) && "i exceeds number of constraints!");
    return storage_[j];
  }

  const ConstraintType &operator[](std::size_t j) const {
    assert((j < this->storage_.size()) && "i exceeds number of constraints!");
    return storage_[j];
  }

protected:
  std::vector<ConstraintType> storage_;
  std::vector<int> cursors_;
  std::vector<int> dims_;
  int total_dim = 0;
};

} // namespace proxddp

#include "proxddp/core/constraint.hxx"
