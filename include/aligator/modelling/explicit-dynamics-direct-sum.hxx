/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, 2023-2025 INRIA
#pragma once

#include "explicit-dynamics-direct-sum.hpp"

namespace aligator {

template <typename Scalar>
DirectSumExplicitDynamicsTpl<Scalar>::DirectSumExplicitDynamicsTpl(
    xyz::polymorphic<Base> f, xyz::polymorphic<Base> g)
    : Base(xyz::polymorphic<Manifold>{f->space_next_ * g->space_next_},
           f->nu + g->nu)
    , f_(f)
    , g_(g) {
  product_space_ = dynamic_cast<CartesianProduct &>(*this->space_next_);
}

template <typename Scalar>
DirectSumExplicitDynamicsTpl<Scalar>::Data::Data(
    const DirectSumExplicitDynamicsTpl &model)
    : BaseData(model)
    , data1_(model.f_->createData())
    , data2_(model.g_->createData()) {}

template <typename Scalar>
void DirectSumExplicitDynamicsTpl<Scalar>::forward(const ConstVectorRef &x,
                                                   const ConstVectorRef &u,
                                                   BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const CartesianProduct &s = this->product_space_;
  auto xs = s.split(x);
  ConstVectorRef u1 = u.head(f_->nu);
  ConstVectorRef u2 = u.tail(g_->nu);

  f_->forward(xs[0], u1, *d.data1_);
  g_->forward(xs[1], u2, *d.data2_);

  d.xnext_.head(s.getComponent(0).nx()) = d.data1_->xnext_;
  d.xnext_.tail(s.getComponent(1).nx()) = d.data2_->xnext_;
}

template <typename Scalar>
void DirectSumExplicitDynamicsTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                    const ConstVectorRef &u,
                                                    BaseData &data) const {
  int nu_f = f_->nu;
  int nu_g = g_->nu;
  int ndxout_f = f_->ndx2();
  int ndxout_g = g_->ndx2();
  Data &d = static_cast<Data &>(data);
  const CartesianProduct &s = this->product_space_;
  auto xs = s.split(x);
  ConstVectorRef x1 = xs[0];
  ConstVectorRef x2 = xs[1];

  ConstVectorRef u1 = u.head(nu_f);
  ConstVectorRef u2 = u.tail(nu_g);

  f_->dForward(x1, u1, *d.data1_);
  g_->dForward(x2, u2, *d.data2_);

  const Manifold &s1 = s.getComponent(0);
  const Manifold &s2 = s.getComponent(1);

  d.Jx().topLeftCorner(ndxout_f, s1.ndx()) = d.data1_->Jx();
  d.Jx().bottomRightCorner(ndxout_g, s2.ndx()) = d.data2_->Jx();

  d.Ju().topLeftCorner(ndxout_f, nu_f) = d.data1_->Ju();
  d.Ju().bottomRightCorner(ndxout_g, nu_g) = d.data2_->Ju();
}

} // namespace aligator
