/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/constraint-set.hpp"

namespace aligator {

///
/// @brief   Negative orthant, for constraints \f$h(x)\leq 0\f$.
///
/// Negative orthant, corresponding to constraints of the form
/// \f[
///    h(x) \leq 0
/// \f]
/// where \f$h : \mathcal{X} \to \RR^p\f$ is a given residual.
///
template <typename _Scalar>
struct NegativeOrthantTpl : ConstraintSetTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  NegativeOrthantTpl() = default;
  NegativeOrthantTpl(const NegativeOrthantTpl &) = default;
  NegativeOrthantTpl &operator=(const NegativeOrthantTpl &) = default;
  NegativeOrthantTpl(NegativeOrthantTpl &&) = default;
  NegativeOrthantTpl &operator=(NegativeOrthantTpl &&) = default;

  using Base = ConstraintSetTpl<Scalar>;
  using ActiveType = typename Base::ActiveType;

  void projection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z.cwiseMin(static_cast<Scalar>(0.));
  }

  void normalConeProjection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z.cwiseMax(static_cast<Scalar>(0.));
  }

  /// The elements of the active set are the components such that \f$z_i > 0\f$.
  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out.array() = (z.array() > static_cast<Scalar>(0.));
  }
};

} // namespace aligator
