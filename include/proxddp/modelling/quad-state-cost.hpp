/// @file
/// Convenience classes to define quadratic state or control cost functions.
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/composite-costs.hpp"

namespace proxddp {

/// Quadratic distance cost over the state manifold.
template <typename Scalar>
struct QuadraticStateCostTpl : QuadraticResidualCostTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = QuadraticResidualCostTpl<Scalar>;
  using Error = StateErrorResidualTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  QuadraticStateCostTpl(shared_ptr<Error> resdl, const MatrixXs &weights)
      : Base(resdl, weights) {}

  QuadraticStateCostTpl(const shared_ptr<Manifold> &xspace, const int nu,
                        const ConstVectorRef &target, const MatrixXs &weights)
      : QuadraticStateCostTpl(std::make_shared<Error>(xspace, nu, target),
                              weights) {}

  void setTarget(const ConstVectorRef target) { residual().target_ = target; }
  ConstVectorRef getTarget() const { return residual().target_; }

protected:
  Error &residual() { return static_cast<Error &>(*this->residual_); }
  const Error &residual() const {
    return static_cast<const Error &>(*this->residual_);
  }
};

template <typename Scalar>
struct QuadraticControlCostTpl : QuadraticResidualCostTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = QuadraticResidualCostTpl<Scalar>;
  using Error = ControlErrorResidualTpl<Scalar>;

  QuadraticControlCostTpl(shared_ptr<Error> resdl, const MatrixXs &weights)
      : Base(resdl, weights) {}

  QuadraticControlCostTpl(int ndx, int nu, const MatrixXs &weights)
      : QuadraticControlCostTpl(std::make_shared<Error>(ndx, nu), weights) {}

  QuadraticControlCostTpl(int ndx, const ConstVectorRef target,
                          const MatrixXs &weights)
      : QuadraticControlCostTpl(std::make_shared<Error>(ndx, target), weights) {
  }

  void setTarget(const ConstVectorRef target) { residual().target_ = target; }
  ConstVectorRef getTarget() const { return residual().target_; }

protected:
  Error &residual() { return static_cast<Error &>(*this->residual_); }
  const Error &residual() const {
    return static_cast<const Error &>(*this->residual_);
  }
};

} // namespace proxddp
