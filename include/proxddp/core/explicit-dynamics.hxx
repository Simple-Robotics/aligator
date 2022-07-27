#pragma once

#include "proxddp/core/explicit-dynamics.hpp"

namespace proxddp {
template <typename Scalar>
ExplicitDynamicsModelTpl<Scalar>::ExplicitDynamicsModelTpl(
    const int ndx1, const int nu, const shared_ptr<Manifold> &next_state)
    : Base(ndx1, nu, next_state->ndx()), next_state_(next_state),
      nx2(next_state->nx()) {}

template <typename Scalar>
ExplicitDynamicsModelTpl<Scalar>::ExplicitDynamicsModelTpl(
    const shared_ptr<Manifold> &next_state, const int nu)
    : ExplicitDynamicsModelTpl(next_state->ndx(), nu, next_state) {}

template <typename Scalar>
void ExplicitDynamicsModelTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                const ConstVectorRef &u,
                                                const ConstVectorRef &y,
                                                BaseData &data) const {
  // Call the forward dynamics and set the function residual
  // value to the difference between y and the xnext_.
  Data &d = static_cast<Data &>(data);
  this->forward(x, u, d);
  next_state_->difference(y, d.xnext_, d.value_);
}

template <typename Scalar>
void ExplicitDynamicsModelTpl<Scalar>::computeJacobians(const ConstVectorRef &x,
                                                        const ConstVectorRef &u,
                                                        const ConstVectorRef &y,
                                                        BaseData &data) const {
  Data &data_ = static_cast<Data &>(data);
  this->forward(x, u, data_);
  this->dForward(x, u, data_);
  // compose by jacobians of log (xout - y)
  next_state_->Jdifference(y, data_.xnext_, data_.Jy_, 0);
  next_state_->Jdifference(y, data_.xnext_, data_.Jtmp_xnext, 1);
  data_.Jx_ = data_.Jtmp_xnext * data_.Jx_;
  data_.Ju_ = data_.Jtmp_xnext * data_.Ju_;
}

template <typename Scalar>
shared_ptr<DynamicsDataTpl<Scalar>>
ExplicitDynamicsModelTpl<Scalar>::createData() const {
  return std::make_shared<Data>(this->ndx1, this->nu, this->nx2, this->ndx2);
}

template <typename Scalar>
ExplicitDynamicsDataTpl<Scalar>::ExplicitDynamicsDataTpl(const int ndx1,
                                                         const int nu,
                                                         const int nx2,
                                                         const int ndx2)
    : Base(ndx1, nu, ndx2, ndx2), xnext_(nx2), dx_(ndx2),
      Jtmp_xnext(ndx2, ndx2), xnextref_(xnext_), dxref_(dx_) {
  xnext_.setZero();
  dx_.setZero();
  Jtmp_xnext.setZero();
}

} // namespace proxddp
