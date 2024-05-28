/// @file constraint.hpp
/// @brief Defines the constraint object and constraint stack manager for this
/// library.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"

#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>

namespace aligator {

/// @brief Simple struct holding together a function and set, to describe a
/// constraint.
template <typename Scalar> struct StageConstraintTpl {
  xyz::polymorphic<StageFunctionTpl<Scalar>> func;
  xyz::polymorphic<ConstraintSetBase<Scalar>> set;
  long nr() const { return func->nr; }
};

/// @brief Convenience class to manage a stack of constraints.
template <typename Scalar> struct ConstraintStackTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintType = StageConstraintTpl<Scalar>;
  using value_type = ConstraintType;
  using data_type = ConstraintType;
  using iterator = typename std::vector<ConstraintType>::iterator;

  ConstraintStackTpl() : indices_({0}) {};

  auto begin() { return storage_.begin(); }
  auto end() { return storage_.end(); }

  std::size_t size() const { return storage_.size(); }
  bool empty() const { return size() == 0; }
  void clear();

  void pushBack(const ConstraintType &el, const long nr);
  void pushBack(const ConstraintType &el);

  /// @brief Get the set of dimensions for each constraint in the stack.
  const std::vector<long> &dims() const { return dims_; }
  /// @copybrief getDims()
  ALIGATOR_DEPRECATED const std::vector<long> &getDims() const { return dims_; }

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

#include "aligator/core/constraint.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/constraint.txx"
#endif
