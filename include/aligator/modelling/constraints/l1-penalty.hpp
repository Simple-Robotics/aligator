/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/constraint-set.hpp"

namespace aligator {

///
/// @brief   Composite \f$\ell_1\f$-penalty function \f$ \|c(x)\|_1 \f$.
///
/// @details The composite \f$\ell_1\f$-penalty penalizes the norm
///          \f$ \| r(x) \|_1\f$ of a residual function.
///          This class implements the proximity operator (soft-thresholding)
///          and an appropriate generalized Jacobian.
///
template <typename _Scalar>
struct NonsmoothPenaltyL1Tpl : ConstraintSetTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  NonsmoothPenaltyL1Tpl() = default;
  NonsmoothPenaltyL1Tpl(const NonsmoothPenaltyL1Tpl &) = default;
  NonsmoothPenaltyL1Tpl &operator=(const NonsmoothPenaltyL1Tpl &) = default;
  NonsmoothPenaltyL1Tpl(NonsmoothPenaltyL1Tpl &&) = default;
  NonsmoothPenaltyL1Tpl &operator=(NonsmoothPenaltyL1Tpl &&) = default;

  using Base = ConstraintSetTpl<Scalar>;
  using ActiveType = typename Base::ActiveType;
  using Base::mu_;

  Scalar evaluate(const ConstVectorRef &zproj) const {
    return zproj.template lpNorm<1>();
  }

  decltype(auto) projection_impl(const ConstVectorRef &z) const {
    return z.array().sign() *
           (z.array().abs() - mu_).max(static_cast<Scalar>(0.));
  }

  void projection(const ConstVectorRef &z, VectorRef zout) const {
    zout = projection_impl(z);
  }

  void normalConeProjection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z - projection_impl(z).matrix();
  }

  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out = z.array().abs() <= mu_;
  }
};

} // namespace aligator
