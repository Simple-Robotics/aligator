/// @file
/// Convenience classes to define quadratic state or control cost functions.
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/state-error.hpp"
#include "aligator/modelling/composite-costs.hpp"

namespace aligator {

/// Quadratic distance cost over the state manifold.
template <typename Scalar>
struct QuadraticStateCostTpl : QuadraticResidualCostTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = QuadraticResidualCostTpl<Scalar>;
  using StateError = StateErrorResidualTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  // StateError's space variable holds a pointer to the state manifold
  QuadraticStateCostTpl(shared_ptr<StateError> resdl, const MatrixXs &weights)
      : Base(resdl->space_, resdl, weights) {}

  QuadraticStateCostTpl(shared_ptr<Manifold> space, const int nu,
                        const ConstVectorRef &target, const MatrixXs &weights)
      : QuadraticStateCostTpl(std::make_shared<StateError>(space, nu, target),
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
  using Error = ControlErrorResidualTpl<Scalar>;

  QuadraticControlCostTpl(shared_ptr<Manifold> space, shared_ptr<Error> resdl,
                          const MatrixXs &weights)
      : Base(space, resdl, weights) {}

  QuadraticControlCostTpl(shared_ptr<Manifold> space, int nu,
                          const ConstMatrixRef &weights)
      : QuadraticControlCostTpl(
            space, std::make_shared<Error>(space->ndx(), nu), weights) {}

  QuadraticControlCostTpl(shared_ptr<Manifold> space,
                          const ConstVectorRef &target,
                          const ConstMatrixRef &weights)
      : QuadraticControlCostTpl(
            space, std::make_shared<Error>(space->ndx(), target), weights) {}

  void setTarget(const ConstVectorRef &target) { residual().target_ = target; }
  ConstVectorRef getTarget() const { return residual().target_; }

protected:
  Error &residual() { return static_cast<Error &>(*this->residual_); }
  const Error &residual() const {
    return static_cast<const Error &>(*this->residual_);
  }
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/quad-state-cost.txx"
#endif
