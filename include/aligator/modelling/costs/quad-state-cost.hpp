/// @file
/// Convenience classes to define quadratic state or control cost functions.
/// @copyright Copyright (C) 2023 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "aligator/modelling/state-error.hpp"
#include "aligator/modelling/costs/quad-residual-cost.hpp"

namespace aligator {

/// Quadratic distance cost over the state manifold.
template <typename Scalar>
struct QuadraticStateCostTpl : QuadraticResidualCostTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = QuadraticResidualCostTpl<Scalar>;
  using StateError = StateErrorResidualTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;

  // StateError's space variable holds a pointer to the state manifold
  QuadraticStateCostTpl(const StateError &resdl, const MatrixXs &weights)
      : Base(resdl.space_, xyz::polymorphic<StageFunction>(resdl), weights) {}

  QuadraticStateCostTpl(xyz::polymorphic<Manifold> space, const int nu,
                        const ConstVectorRef &target, const MatrixXs &weights)
      : QuadraticStateCostTpl(StateError(std::move(space), nu, target),
                              weights) {}

  template <typename U,
            typename = std::enable_if_t<!is_polymorphic_of_v<Manifold, U>>>
  QuadraticStateCostTpl(U &&space, const int nu, const ConstVectorRef &target,
                        const MatrixXs &weights)
      : QuadraticStateCostTpl(StateError(std::forward<U>(space), nu, target),
                              weights) {}

  void setTarget(const ConstVectorRef target) { residual().target_ = target; }
  ConstVectorRef getTarget() const { return residual().target_; }

protected:
  StateError &residual() { return static_cast<StateError &>(*this->residual_); }
  const StateError &residual() const {
    return static_cast<const StateError &>(*this->residual_);
  }
};

template <typename Scalar>
struct QuadraticControlCostTpl : QuadraticResidualCostTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = QuadraticResidualCostTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using ControlError = ControlErrorResidualTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;

  QuadraticControlCostTpl(xyz::polymorphic<Manifold> space,
                          const ControlError &resdl, const MatrixXs &weights)
      : Base(std::move(space), xyz::polymorphic<StageFunction>(resdl),
             weights) {}

  QuadraticControlCostTpl(xyz::polymorphic<Manifold> space, int nu,
                          const ConstMatrixRef &weights)
      : QuadraticControlCostTpl(space, ControlError(space->ndx(), nu),
                                weights) {}

  QuadraticControlCostTpl(xyz::polymorphic<Manifold> space,
                          const ConstVectorRef &target,
                          const ConstMatrixRef &weights)
      : QuadraticControlCostTpl(space, ControlError(space->ndx(), target),
                                weights) {}

  void setTarget(const ConstVectorRef &target) { residual().target_ = target; }
  ConstVectorRef getTarget() const { return residual().target_; }

protected:
  ControlError &residual() {
    return static_cast<ControlError &>(*this->residual_);
  }
  const ControlError &residual() const {
    return static_cast<const ControlError &>(*this->residual_);
  }
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./quad-state-cost.txx"
#endif
