/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/constraint-set.hpp"

namespace aligator {

/// @brief   Box constraint set \f$z \in [z_\min, z_\max]\f$.
template <typename Scalar> struct BoxConstraintTpl : ConstraintSetTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ConstraintSetTpl<Scalar>;
  using ActiveType = typename Base::ActiveType;

  VectorXs lower_limit;
  VectorXs upper_limit;

  BoxConstraintTpl(const ConstVectorRef lower, const ConstVectorRef upper)
      : Base(), lower_limit(lower), upper_limit(upper) {}
  BoxConstraintTpl(const BoxConstraintTpl &) = default;
  BoxConstraintTpl &operator=(const BoxConstraintTpl &) = default;
  BoxConstraintTpl(BoxConstraintTpl &&) = default;
  BoxConstraintTpl &operator=(BoxConstraintTpl &&) = default;

  decltype(auto) projection_impl(const ConstVectorRef &z) const {
    return z.cwiseMin(upper_limit).cwiseMax(lower_limit);
  }

  void projection(const ConstVectorRef &z, VectorRef zout) const {
    zout = projection_impl(z);
  }

  void normalConeProjection(const ConstVectorRef &z, VectorRef zout) const {
    zout = z - projection_impl(z);
  }

  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const {
    out.array() =
        (z.array() > upper_limit.array()) || (z.array() < lower_limit.array());
  }
};

} // namespace aligator
