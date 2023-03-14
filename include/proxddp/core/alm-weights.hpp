#pragma once

#include "proxddp/math.hpp"

namespace proxddp {

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

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/context.hpp"

namespace proxddp {
extern template struct ConstraintALWeightStrategy<context::Scalar>;
}
#endif
