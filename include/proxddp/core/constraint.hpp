/// @file constraint.hpp
/// @brief Defines the constraint object for this library.
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {

/// @brief Simple struct holding together a function and set, to describe a
/// constraint.
template <typename Scalar> struct StageConstraintTpl {
  shared_ptr<StageFunctionTpl<Scalar>> func_;
  shared_ptr<ConstraintSetBase<Scalar>> set_;
};

/// @brief Convenience class to manage a stack of constraints.
template <typename Scalar> struct ConstraintContainer {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintType = StageConstraintTpl<Scalar>;
  ConstraintContainer() : cursors_({0}){};

  std::size_t numConstraints() const { return storage_.size(); }

  void push_back(const ConstraintType &el) {
    assert(el.func_ != 0 && "member func can't be called with nullptr");
    this->push_back(el, el.func_->nr);
  }

  void push_back(const ConstraintType &el, const int nr) {
    const int last_cursor = cursors_.back();
    storage_.push_back(el);
    cursors_.push_back(last_cursor + nr);
    dims_.push_back(nr);
    total_dim += nr;
  }

  int getIndex(const std::size_t i) const { return cursors_[i]; }

  int getDim(const std::size_t i) const { return dims_[i]; }

  const ConstraintSetBase<Scalar> &getConstraintSet(const std::size_t i) const {
    return *this->storage_[i].set_;
  }

  /// Get corresponding segment of a vector corresponding
  /// to the @p i-th constraint.
  template <typename Derived>
  Eigen::VectorBlock<Derived, -1>
  getSegmentByConstraint(const Eigen::MatrixBase<Derived> &lambda,
                         const std::size_t i) const {
    using MatrixType = Eigen::MatrixBase<Derived>;
    MatrixType &lam_cast = const_cast<MatrixType &>(lambda);
    assert(lambda.size() == totalDim());
    return lam_cast.segment(getIndex(i), getDim(i));
  }

  template <typename Derived>
  Eigen::VectorBlock<const Derived, -1>
  getConstSegmentByConstraint(const Eigen::MatrixBase<Derived> &lambda,
                              const std::size_t i) const {
    assert(lambda.size() == totalDim());
    return lambda.segment(getIndex(i), getDim(i));
  }

  /// Get a row-wise block of a matrix by constraint index.
  template <typename Derived>
  Eigen::Block<Derived, -1, -1>
  getBlockByConstraint(const Eigen::MatrixBase<Derived> &J_,
                       const std::size_t i) const {
    using M = Eigen::MatrixBase<Derived>;
    M &J = const_cast<M &>(J_);
    assert(J.rows() == totalDim());
    return J.middleRows(getIndex(i), getDim(i));
  }

  int totalDim() const { return total_dim; }

  /// Get the i-th constraint.
  ConstraintType &operator[](std::size_t i) {
    assert((i < this->storage_.size()) && "i exceeds number of constraints!");
    return storage_[i];
  }

  const ConstraintType &operator[](std::size_t i) const {
    assert((i < this->storage_.size()) && "i exceeds number of constraints!");
    return storage_[i];
  }

protected:
  std::vector<ConstraintType> storage_;
  std::vector<int> cursors_;
  std::vector<int> dims_;
  int total_dim = 0;
};

} // namespace proxddp
