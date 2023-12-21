#pragma once

#include "proxddp/core/cost-abstract.hpp"

namespace aligator {

/// @brief Constant cost.
template <typename _Scalar> struct ConstantCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  Scalar value_;
  ConstantCostTpl(shared_ptr<Manifold> space, const int nu, const Scalar value)
      : Base(space, nu), value_(value) {}
  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                CostData &data) const override {
    data.value_ = value_;
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        CostData &) const override {}

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &) const override {}

  shared_ptr<CostData> createData() const override {
    auto d = Base::createData();
    d->value_ = value_;
    return d;
  }
};

} // namespace aligator
