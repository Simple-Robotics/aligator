/// @file constraint.hpp
/// @brief Defines the constraint object for this library.
#pragma once

#include "proxddp/core/function.hpp"

namespace proxddp {

/** @brief  Base class for stage-wise constraint objects.
 *
 * This class packs a StageFunctionTpl and ConstraintSetBase together.
 * It models stage-wise constraints of the form
 * \f[
 *        c(x, u, x') \in \mathcal{C}.
 * \f]
 */
template <typename _Scalar> struct StageConstraintTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  using FunctionType = StageFunctionTpl<Scalar>;
  using ConstraintSetPtr = shared_ptr<ConstraintSetBase<Scalar>>;
  using FunctionPtr = shared_ptr<const FunctionType>;

  const FunctionPtr func_;
  ConstraintSetPtr set_;

  StageConstraintTpl(const FunctionPtr &func,
                     const ConstraintSetPtr &constraint_set)
      : func_(func), set_(constraint_set) {}

  inline int nr() const { return func_->nr; }

  const ConstraintSetBase<Scalar> &getConstraintSet() const { return *set_; }

  const FunctionType &func() const { return *func_; }
};

/// @brief Convenience class to manage a stack of constraints.
template <typename Scalar> struct ConstraintContainer {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Constraint = StageConstraintTpl<Scalar>;
  ConstraintContainer() : cursors_({0}){};

  std::size_t numConstraints() const { return storage_.size(); }

  void push_back(const shared_ptr<Constraint> &el) {
    const int nr = el->nr();
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
  Eigen::VectorBlock<VectorRef, -1>
  getSegmentByConstraint(VectorRef lambda, const std::size_t i) const {
    assert(lambda.size() == totalDim());
    return lambda.segment(getIndex(i), getDim(i));
  }

  ConstVectorRef getConstSegmentByConstraint(const ConstVectorRef lambda,
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
  shared_ptr<Constraint> &operator[](std::size_t i) { return storage_[i]; }

  const shared_ptr<Constraint> &operator[](std::size_t i) const {
    return storage_[i];
  }

protected:
  std::vector<shared_ptr<Constraint>> storage_;
  std::vector<int> cursors_;
  std::vector<int> dims_;
  int total_dim = 0;
};

} // namespace proxddp
