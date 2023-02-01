/// @file constraint.hpp
/// @brief Defines the constraint object and constraint stack manager for this
/// library.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
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
  ConstraintStackTpl() : indices_({0}){};

  std::size_t size() const { return storage_.size(); }
  bool empty() const { return size() == 0; }
  void clear() {
    storage_.clear();
    indices_ = {0};
    dims_.clear();
    total_dim_ = 0;
  }

  void pushBack(const ConstraintType &el, const long nr);
  void pushBack(const ConstraintType &el);

  /// @brief Get start index in an array.
  long getIndex(const std::size_t j) const { return indices_[j]; }

  /// @brief Get the dimension of each the @p j-th constraint.
  long getDim(const std::size_t j) const { return dims_[j]; }

  /// @brief Get the set of dimensions for each constraint in the stack.
  const std::vector<long> &getDims() const { return dims_; }

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

/// @brief  Weighting strategy for the constraints in a stack.
template <typename Scalar> struct ConstraintALWeightStrategy {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  ConstraintALWeightStrategy(Scalar mu, bool weighted)
      : mu_(mu), dyn_weight_(1e-3), weighted_(weighted) {}

  Scalar get(std::size_t j) {
    if (!weighted_)
      return mu_;
    else if (j == 0) {
      return dyn_weight_ * mu_;
    } else {
      return mu_;
    }
  }

  Scalar inv(std::size_t j) { return 1. / get(j); }

  inline void enable() { weighted_ = true; }
  inline void disable() { weighted_ = false; }

private:
  Scalar mu_;
  Scalar dyn_weight_; // weighting for dynamical constraints (index j == 0)
  bool weighted_;     // whether weighting is activated
};

} // namespace proxddp

#include "proxddp/core/constraint.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/constraint.txx"
#endif
