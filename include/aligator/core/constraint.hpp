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
};

/// @brief Convenience class to manage a stack of constraints.
template <typename Scalar> struct ConstraintStackTpl {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintType = StageConstraintTpl<Scalar>;
  using value_type = ConstraintType;
  using data_type = ConstraintType;
  using PolyFunc = xyz::polymorphic<StageFunctionTpl<Scalar>>;
  using PolySet = xyz::polymorphic<ConstraintSetBase<Scalar>>;

  ConstraintStackTpl() : indices_({0}) {};
  ConstraintStackTpl(const ConstraintStackTpl &) = default;
  ConstraintStackTpl &operator=(const ConstraintStackTpl &) = default;
  ConstraintStackTpl(ConstraintStackTpl &&) = default;
  ConstraintStackTpl &operator=(ConstraintStackTpl &&) = default;

  std::size_t size() const { return funcs.size(); }
  bool empty() const { return size() == 0; }
  void clear();

  template <typename Cstr> void pushBack(Cstr &&el) {
    assert(!el.func.valueless_after_move() &&
           "constraint must have non-null underlying function.");
    assert(!el.set.valueless_after_move() &&
           "constraint must have non-null underlying set.");
    funcs.emplace_back(el.func);
    sets.emplace_back(el.set);
    addDim(el.func->nr);
  }

  void pushBack(const PolyFunc &func, const PolySet &cstr_set) {
    assert(!func.valueless_after_move() &&
           "constraint must have non-null underlying function.");
    assert(!cstr_set.valueless_after_move() &&
           "constraint must have non-null underlying set.");
    funcs.emplace_back(func);
    sets.emplace_back(cstr_set);
    addDim(func->nr);
  }

  /// @brief Get the set of dimensions for each constraint in the stack.
  const std::vector<long> &dims() const { return dims_; }
  /// @copybrief getDims()
  ALIGATOR_DEPRECATED const std::vector<long> &getDims() const { return dims_; }

  long totalDim() const { return total_dim_; }

  /// @brief Get constraint function, cast down to the specified type.
  template <typename Derived> Derived *getConstraint(const size_t id) {
    return dynamic_cast<Derived *>(&*funcs[id]);
  }

  /// @copybrief getConstraint()
  template <typename Derived>
  const Derived *getConstraint(const size_t id) const {
    return dynamic_cast<const Derived *>(&*funcs[id]);
  }

  std::vector<PolyFunc> funcs;
  std::vector<PolySet> sets;

protected:
  std::vector<long> indices_;
  std::vector<long> dims_;
  long total_dim_ = 0;

private:
  void addDim(const long nr) {
    const long last_cursor = indices_.back();
    indices_.push_back(last_cursor + nr);
    dims_.push_back(nr);
    total_dim_ += nr;
  }
};

} // namespace aligator

#include "aligator/core/constraint.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/constraint.txx"
#endif
