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
  using Constraint = StageConstraintTpl<Scalar>;
  ConstraintContainer() : cursors_({0}){};

  std::size_t numConstraints() const { return storage_.size(); }

  void push_back(const Constraint &el) {
    const int nr = el.func_->nr;
    const int last_cursor = cursors_.back();
    storage_.push_back(el);
    cursors_.push_back(last_cursor + nr);
    dims_.push_back(nr);
    total_dim += nr;
  }

  int getIndex(const std::size_t i) const { return cursors_[i]; }

  int getDim(const std::size_t i) const { return dims_[i]; }

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

  Eigen::Block<MatrixRef, -1, -1>
  getBlockByConstraint(MatrixRef J, const std::size_t i) const {
    assert(J.rows() == totalDim());
    return J.middleRows(getIndex(i), getDim(i));
  }

  int totalDim() const { return total_dim; }

  /// Get the i-th constraint.
  Constraint &operator[](std::size_t i) { return storage_[i]; }

  const Constraint &operator[](std::size_t i) const { return storage_[i]; }

protected:
  std::vector<Constraint> storage_;
  std::vector<int> cursors_;
  std::vector<int> dims_;
  int total_dim = 0;
};

} // namespace proxddp
