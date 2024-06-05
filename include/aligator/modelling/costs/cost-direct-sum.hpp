/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/cost-abstract.hpp"
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>

namespace aligator {

template <typename _Scalar> struct DirectSumCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using BaseCost = CostAbstractTpl<Scalar>;
  using BaseData = CostDataAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  struct Data;

  DirectSumCostTpl(shared_ptr<BaseCost> c1, shared_ptr<BaseCost> c2)
      : BaseCost(c1->space * c2->space, c1->nu + c2->nu), c1_(c1), c2_(c2) {
    assert(c1 != nullptr && c2 != nullptr);
  }

  shared_ptr<BaseCost> c1_, c2_;

  shared_ptr<BaseData> createData() const override;
  shared_ptr<BaseData>
  createData(const CommonModelDataContainer &container) const override;

  void configure(
      CommonModelBuilderContainer &common_buider_container) const override;
  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override;
  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const override;
  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const override;

private:
  using CartesianProduct = proxsuite::nlp::CartesianProductTpl<Scalar>;
  auto get_product_space() const {
    return static_cast<CartesianProduct const *>(this->space.get());
  }
  static Data &data_cast(BaseData &data) { return static_cast<Data &>(data); }
};

template <typename Scalar> struct DirectSumCostTpl<Scalar>::Data : BaseData {

  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  shared_ptr<BaseData> data1_, data2_;
  Data(const DirectSumCostTpl &model)
      : BaseData(model.ndx(), model.nu), data1_(model.c1_->createData()),
        data2_(model.c2_->createData()) {}

  Data(const DirectSumCostTpl &model, const CommonModelDataContainer &container)
      : BaseData(model.ndx(), model.nu),
        data1_(model.c1_->createData(container)),
        data2_(model.c2_->createData(container)) {}
};

template <typename Scalar>
void DirectSumCostTpl<Scalar>::configure(
    CommonModelBuilderContainer &common_buider_container) const {
  c1_->configure(common_buider_container);
  c2_->configure(common_buider_container);
}

template <typename Scalar>
void DirectSumCostTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                        const ConstVectorRef &u,
                                        BaseData &data) const {
  CartesianProduct const *space = get_product_space();
  Data &d = data_cast(data);
  auto xs = space->split(x);
  ConstVectorRef u1 = u.head(c1_->nu);
  ConstVectorRef u2 = u.tail(c2_->nu);

  c1_->evaluate(xs[0], u1, *d.data1_);
  c2_->evaluate(xs[1], u2, *d.data2_);

  d.value_ = d.data1_->value_ + d.data2_->value_;
}

template <typename Scalar>
void DirectSumCostTpl<Scalar>::computeGradients(const ConstVectorRef &x,
                                                const ConstVectorRef &u,
                                                BaseData &data) const {
  CartesianProduct const *space = get_product_space();
  Data &d = data_cast(data);
  auto xs = space->split(x);
  ConstVectorRef u1 = u.head(c1_->nu);
  ConstVectorRef u2 = u.tail(c2_->nu);

  BaseData &d1 = *d.data1_;
  BaseData &d2 = *d.data2_;

  c1_->computeGradients(xs[0], u1, d1);
  c2_->computeGradients(xs[1], u2, d2);

  d.Lx_.head(c1_->ndx()) = d1.Lx_;
  d.Lx_.tail(c2_->ndx()) = d2.Lx_;
  d.Lu_.head(c1_->nu) = d1.Lu_;
  d.Lu_.tail(c2_->nu) = d2.Lu_;
}

template <typename Scalar>
void DirectSumCostTpl<Scalar>::computeHessians(const ConstVectorRef &x,
                                               const ConstVectorRef &u,
                                               BaseData &data) const {
  CartesianProduct const *space = get_product_space();
  Data &d = data_cast(data);
  auto xs = space->split(x);
  ConstVectorRef u1 = u.head(c1_->nu);
  ConstVectorRef u2 = u.tail(c2_->nu);

  BaseData &d1 = *d.data1_;
  BaseData &d2 = *d.data2_;

  c1_->computeHessians(xs[0], u1, d1);
  c2_->computeHessians(xs[1], u2, d2);

  d.Lxx_.topLeftCorner(c1_->ndx(), c1_->ndx()) = d1.Lxx_;
  d.Lxx_.bottomRightCorner(c2_->ndx(), c2_->ndx()) = d2.Lxx_;

  d.Luu_.topLeftCorner(c1_->nu, c1_->nu) = d1.Luu_;
  d.Luu_.bottomRightCorner(c2_->nu, c2_->nu) = d2.Luu_;

  d.Lxu_.topLeftCorner(c1_->ndx(), c1_->nu) = d1.Lxu_;
  d.Lxu_.bottomRightCorner(c2_->ndx(), c2_->nu) = d2.Lxu_;

  d.Lux_ = d.Lxu_.transpose();
}

template <typename Scalar>
auto DirectSumCostTpl<Scalar>::createData() const -> shared_ptr<BaseData> {
  return std::make_shared<Data>(*this);
}

template <typename Scalar>
auto DirectSumCostTpl<Scalar>::createData(
    const CommonModelDataContainer &container) const -> shared_ptr<BaseData> {
  return std::make_shared<Data>(*this, container);
}

template <typename Scalar>
auto directSum(shared_ptr<CostAbstractTpl<Scalar>> const &c1,
               shared_ptr<CostAbstractTpl<Scalar>> const &c2) {
  return std::make_shared<DirectSumCostTpl<Scalar>>(c1, c2);
}

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./cost-direct-sum.txx"
#endif
