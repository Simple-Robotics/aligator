/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/cost-abstract.hpp"
#include <proxnlp/modelling/spaces/cartesian-product.hpp>

namespace proxddp {

template <typename _Scalar> struct DirectSumCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BaseCost = CostAbstractTpl<Scalar>;
  using BaseData = CostDataAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  struct Data;

  DirectSumCostTpl(shared_ptr<BaseCost> c1, shared_ptr<BaseCost> c2)
      : BaseCost(c1->space * c2->space, c1->nu + c2->nu), c1_(c1), c2_(c2) {
    assert(c1 != nullptr && c2 != nullptr);
  }

  shared_ptr<BaseCost> c1_, c2_;

  shared_ptr<BaseData> createData() const override;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override {
    CartesianProduct const *space = get_product_space();
  }
  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const override {
    auto space = get_product_space();
  }
  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const override {}

private:
  using CartesianProduct = proxnlp::CartesianProductTpl<Scalar>;
  auto get_product_space() const {
    return static_cast<CartesianProduct const *>(this->space.get());
  }
};

template <typename Scalar> struct DirectSumCostTpl<Scalar>::Data : BaseData {

  shared_ptr<BaseData> data1_, data2_;
  Data(const DirectSumCostTpl &model)
      : BaseData(model.ndx(), model.nu), data1_(model.c1_->createData()),
        data2_(model.c2_->createData()) {}
};

template <typename Scalar>
auto DirectSumCostTpl<Scalar>::createData() const -> shared_ptr<BaseData> {
  return std::make_shared<Data>(*this);
}

template <typename Scalar>
auto directSum(shared_ptr<CostAbstractTpl<Scalar>> const &c1,
               shared_ptr<CostAbstractTpl<Scalar>> const &c2) {
  return std::make_shared<DirectSumCostTpl<Scalar>>(c1, c2);
}

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./cost-direct-sum.txx"
#endif
