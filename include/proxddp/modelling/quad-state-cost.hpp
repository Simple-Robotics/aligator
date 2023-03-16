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

  void setTarget(const ConstMatrixRef target) {
    getResidual().target_ = target;
  }

  ConstMatrixRef getTarget() const { return getResidual().target_; }

private:
  Error &getResidual() { return static_cast<Error &>(*this->residual_); }
  const Error &getResidual() const {
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

  template <typename... Args>
  QuadraticControlCostTpl(Args... args, const MatrixXs &weights)
      : Base(std::make_shared<Error>(std::forward<Args>(args)...), weights) {}
};

} // namespace proxddp
