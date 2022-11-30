#pragma once

#include "proxddp/core/cost-abstract.hpp"

namespace proxddp {

/// @brief Constant cost.
template <typename _Scalar> struct ConstantCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  Scalar value_;
  ConstantCostTpl(const int ndx, const int nu, const Scalar value)
      : Base(ndx, nu), value_(value) {}
  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                CostData &data) const {
    data.value_ = value_;
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        CostData &) const {}

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &) const {}
};

} // namespace proxddp
