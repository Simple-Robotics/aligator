/// @file constraint.hpp
/// @brief Defines the constraint object and constraint stack manager for this
/// library.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace aligator {

/// @brief Simple struct holding together a function and set, to describe a
/// constraint.
template <typename Scalar> struct StageConstraintTpl {
  shared_ptr<StageFunctionTpl<Scalar>> func;
  shared_ptr<ConstraintSetBase<Scalar>> set;
  long nr() const { return func->nr; }
};

/// @brief Convenience class to manage a stack of constraints.
template <typename Scalar> struct ConstraintStackTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintType = StageConstraintTpl<Scalar>;
  using value_type = ConstraintType;
  using data_type = ConstraintType;
  using iterator = typename std::vector<ConstraintType>::iterator;

  ConstraintStackTpl() : indices_({0}){};

  auto begin() { return storage_.begin(); }
  auto end() { return storage_.end(); }

  std::size_t size() const { return storage_.size(); }
  bool empty() const { return size() == 0; }
  void clear();

  void pushBack(const ConstraintType &el, const long nr);
  void pushBack(const ConstraintType &el);

  /// @brief Get the set of dimensions for each constraint in the stack.
  const std::vector<long> &getDims() const { return dims_; }

  /// @brief Get corresponding segment of a vector corresponding to the @p i-th
  /// constraint.
  template <typename Derived>
  Eigen::VectorBlock<Derived, -1>
  segmentByConstraint(const Eigen::MatrixBase<Derived> &lambda,
                      const std::size_t j) const;

  /// @copybrief getSegmentByConstraint()
  template <typename Derived>
  Eigen::VectorBlock<const Derived, -1>
  constSegmentByConstraint(const Eigen::MatrixBase<Derived> &lambda,
                           const std::size_t j) const;

  /// Get a row-wise block of a matrix by constraint index.
  template <typename Derived>
  auto rowsByConstraint(const Eigen::MatrixBase<Derived> &J_,
                        const std::size_t j) const;

  long totalDim() const { return total_dim_; }

  /// @brief Get the i-th constraint.
  ConstraintType &operator[](std::size_t j) {
    assert((j < this->storage_.size()) && "i exceeds number of constraints!");
    return storage_[j];
  }

  /// @copybrief operator[]()
  const ConstraintType &operator[](std::size_t j) const {
    assert((j < this->storage_.size()) && "i exceeds number of constraints!");
    return storage_[j];
  }

protected:
  std::vector<ConstraintType> storage_;
  std::vector<long> indices_;
  std::vector<long> dims_;
  long total_dim_ = 0;
};

} // namespace aligator

#include "proxddp/core/constraint.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/constraint.txx"
#endif
