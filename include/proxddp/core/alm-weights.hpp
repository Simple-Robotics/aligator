#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {

/// @brief  Weighting strategy for the constraints in a stack.
template <typename Scalar> struct ConstraintProximalScalerTpl {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintStack = ConstraintStackTpl<Scalar>;

  ConstraintProximalScalerTpl(const ConstraintStack &constraints,
                              const Scalar &mu)
      : constraints_(constraints), mu_(mu), weights(constraints.size()) {
    weights.setOnes();
  }

  Scalar get(std::size_t j) const { return weights[(long)j] * mu_; }

  Scalar inv(std::size_t j) const { return 1. / get(j); }
  void set_weight(const Scalar w, long j) {
    assert(j < weights.size());
    weights[j] = w;
  }

  /// For problem stages. Scale down non-dynamical constraints by 100.
  void applyDefaultStrategy();

  const VectorXs &getWeights() const { return weights; }

private:
  const ConstraintStack &constraints_;
  const Scalar &mu_;
  VectorXs weights;
};

template <typename Scalar>
void ConstraintProximalScalerTpl<Scalar>::applyDefaultStrategy() {
  weights[0] = 1e-3;
  for (long j = 1; j < (long)constraints_.size(); j++) {
    weights[j] = 100.;
  }
}

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./alm-weights.txx"
#endif
