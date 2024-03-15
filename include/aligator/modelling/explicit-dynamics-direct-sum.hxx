/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./explicit-dynamics-direct-sum.hpp"
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>

namespace aligator {

template <typename Scalar>
struct DirectSumExplicitDynamicsTpl<Scalar>::Data : BaseData {
  shared_ptr<BaseData> data1_, data2_;

  Data(DirectSumExplicitDynamicsTpl const &model)
      : BaseData(model.ndx1, model.nu, model.nx2(), model.ndx2),
        data1_(std::static_pointer_cast<BaseData>(model.f_->createData())),
        data2_(std::static_pointer_cast<BaseData>(model.g_->createData())) {}

  Data(DirectSumExplicitDynamicsTpl const &model,
       const CommonModelDataContainer &container)
      : BaseData(model.ndx1, model.nu, model.nx2(), model.ndx2),
        data1_(std::static_pointer_cast<BaseData>(
            model.f_->createData(container))),
        data2_(std::static_pointer_cast<BaseData>(
            model.g_->createData(container))) {}
};

template <typename Scalar>
void DirectSumExplicitDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                   const ConstVectorRef &u,
                                                   BaseData &data) const {
  Data &d = data_cast(data);
  const CartesianProduct &s = *this->product_space_;
  auto xs = s.split(x);
  ConstVectorRef x1 = xs[0];
  ConstVectorRef x2 = xs[1];
  ConstVectorRef u1 = u.head(f_->nu);
  ConstVectorRef u2 = u.tail(g_->nu);

  f_->forward(x1, u1, *d.data1_);
  g_->forward(x2, u2, *d.data2_);

  d.xnext_.head(s.getComponent(0).nx()) = d.data1_->xnext_;
  d.xnext_.tail(s.getComponent(1).nx()) = d.data2_->xnext_;

  d.dx_.head(s.getComponent(0).ndx()) = d.data1_->dx_;
  d.dx_.tail(s.getComponent(1).ndx()) = d.data2_->dx_;
}

template <typename Scalar>
void DirectSumExplicitDynamicsTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                    const ConstVectorRef &u,
                                                    BaseData &data) const {
  int nu_f = f_->nu;
  int nu_g = g_->nu;
  int ndxout_f = f_->ndx2;
  int ndxout_g = g_->ndx2;
  Data &d = data_cast(data);
  const CartesianProduct &s = *this->product_space_;
  auto xs = s.split(x);
  ConstVectorRef x1 = xs[0];
  ConstVectorRef x2 = xs[1];

  ConstVectorRef u1 = u.head(nu_f);
  ConstVectorRef u2 = u.tail(nu_g);

  f_->dForward(x1, u1, *d.data1_);
  g_->dForward(x2, u2, *d.data2_);

  const Manifold &s1 = s.getComponent(0);
  const Manifold &s2 = s.getComponent(1);

  d.Jx_.topLeftCorner(ndxout_f, s1.ndx()) = d.data1_->Jx_;
  d.Jx_.bottomRightCorner(ndxout_g, s2.ndx()) = d.data2_->Jx_;

  d.Ju_.topLeftCorner(ndxout_f, nu_f) = d.data1_->Ju_;
  d.Ju_.bottomRightCorner(ndxout_g, nu_g) = d.data2_->Ju_;
}

template <typename Scalar>
auto DirectSumExplicitDynamicsTpl<Scalar>::get_product_space(Base const &f,
                                                             Base const &g) {
  return f.space_next_ * g.space_next_;
}

} // namespace aligator
