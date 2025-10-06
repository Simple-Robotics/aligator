/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/modelling/spaces/cartesian-product.hpp"

namespace aligator {

/// Direct sum of two explicit-dynamics models. This operates on the product
/// space of the two supplied functions. The expression is \f\[
///   (x_{k+1}, z_{k+1}) = (f(x_k, u_k), g(z_k, w_k))
/// \f\]
/// where \f$f,g\f$ are the two components.
template <typename _Scalar>
struct DirectSumExplicitDynamicsTpl : ExplicitDynamicsModelTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitDynamicsModelTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CartesianProduct = aligator::CartesianProductTpl<Scalar>;
  using BaseData = ExplicitDynamicsDataTpl<Scalar>;

  struct Data;

  DirectSumExplicitDynamicsTpl(xyz::polymorphic<Base> f,
                               xyz::polymorphic<Base> g);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const override;

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override;

  shared_ptr<BaseData> createData() const override {
    return std::make_shared<Data>(*this);
  }

  xyz::polymorphic<Base> f_, g_;

private:
  /// pointer to casted cartesian product space; this pointer does not manage
  /// memory
  CartesianProduct product_space_;
};

template <typename Scalar>
struct DirectSumExplicitDynamicsTpl<Scalar>::Data : BaseData {
  shared_ptr<BaseData> data1_, data2_;
  Data(DirectSumExplicitDynamicsTpl const &model);
};

template <typename Scalar>
auto directSum(xyz::polymorphic<ExplicitDynamicsModelTpl<Scalar>> const &m1,
               xyz::polymorphic<ExplicitDynamicsModelTpl<Scalar>> const &m2) {
  return DirectSumExplicitDynamicsTpl<Scalar>(m1, m2);
}

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct DirectSumExplicitDynamicsTpl<context::Scalar>;
#endif
} // namespace aligator
