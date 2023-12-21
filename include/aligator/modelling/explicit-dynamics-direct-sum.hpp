/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/explicit-dynamics.hpp"
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>

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
  using CartesianProduct = proxsuite::nlp::CartesianProductTpl<Scalar>;
  using BaseData = ExplicitDynamicsDataTpl<Scalar>;

  struct Data;

  DirectSumExplicitDynamicsTpl(shared_ptr<Base> f, shared_ptr<Base> g)
      : Base(get_product_space(*f, *g), f->nu + g->nu), f_(f), g_(g) {
    product_space_ = static_cast<CartesianProduct *>(this->space_next_.get());
  }

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const override;

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override;

  shared_ptr<DynamicsDataTpl<Scalar>> createData() const override {
    return std::make_shared<Data>(*this);
  }

  shared_ptr<Base> f_, g_;

private:
  static auto get_product_space(Base const &f, Base const &g);
  static Data &data_cast(BaseData &data) { return static_cast<Data &>(data); }

  /// pointer to casted cartesian product space; this pointer does not manage
  /// memory
  CartesianProduct const *product_space_;
};

template <typename Scalar>
auto directSum(shared_ptr<ExplicitDynamicsModelTpl<Scalar>> const &m1,
               shared_ptr<ExplicitDynamicsModelTpl<Scalar>> const &m2) {
  return std::make_shared<DirectSumExplicitDynamicsTpl<Scalar>>(m1, m2);
}

} // namespace aligator

#include "aligator/modelling/explicit-dynamics-direct-sum.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./explicit-dynamics-direct-sum.txx"
#endif
