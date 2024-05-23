#pragma once

#include "aligator/modelling/autodiff/cost-finite-difference.hpp"

namespace aligator::autodiff {

template <typename Scalar>
CostFiniteDifferenceHelper<Scalar>::CostFiniteDifferenceHelper(
    xyz::polymorphic<CostBase> cost, const Scalar fd_eps)
    : CostBase(cost->space, cost->nu), cost_(cost), fd_eps(fd_eps) {}

template <typename Scalar>
void CostFiniteDifferenceHelper<Scalar>::evaluate(const ConstVectorRef &x,
                                                  const ConstVectorRef &u,
                                                  CostData &data_) const {
  Data &d = static_cast<Data &>(data_);
  cost_->evaluate(x, u, *d.c1);

  d.value_ = d.c1->value_;
}

template <typename Scalar>
void CostFiniteDifferenceHelper<Scalar>::computeGradients(
    const ConstVectorRef &x, const ConstVectorRef &u, CostData &data_) const {
  Data &d = static_cast<Data &>(data_);
  Manifold const &space = *this->space;

  cost_->evaluate(x, u, *d.c1);

  d.dx.setZero();
  for (int i = 0; i < this->ndx(); i++) {
    d.dx[i] = fd_eps;
    space.integrate(x, d.dx, d.xp);
    cost_->evaluate(d.xp, u, *d.c2);

    d.Lx_[i] = (d.c2->value_ - d.c1->value_) / fd_eps;
    d.dx[i] = 0.;
  }

  d.du.setZero();
  for (int i = 0; i < this->nu; i++) {
    d.du[i] = fd_eps;
    d.up = u + d.du;
    cost_->evaluate(x, d.up, *d.c2);

    d.Lu_[i] = (d.c2->value_ - d.c1->value_) / fd_eps;
    d.du[i] = 0.;
  }
}

template <typename Scalar>
auto CostFiniteDifferenceHelper<Scalar>::createData() const
    -> shared_ptr<CostData> {
  return std::make_shared<Data>(*this);
}

} // namespace aligator::autodiff
