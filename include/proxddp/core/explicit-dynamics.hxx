#pragma once

#include "proxddp/core/explicit-dynamics.hpp"


namespace proxddp
{
  template<typename Scalar>
  void ExplicitDynamicsModelTpl<Scalar>::
  evaluate(const ConstVectorRef& x,
           const ConstVectorRef& u,
           const ConstVectorRef& y,
           BaseData& data) const
  {
    // Call the forward dynamics and set the function residual
    // value to the difference between y and the xout_.
    ExplicitData& data_ = static_cast<ExplicitData&>(data);
    this->forward(x, u, data_);
    out_space().difference(y, data_.xout_, data_.value_);  // xnext - y
  }

  template<typename Scalar>
  void ExplicitDynamicsModelTpl<Scalar>::
  computeJacobians(const ConstVectorRef& x,
                   const ConstVectorRef& u,
                   const ConstVectorRef& y,
                   BaseData& data) const
  {
    ExplicitData& data_ = static_cast<ExplicitData&>(data);
    this->forward(x, u, data_);
    this->dForward(x, u, data_); // dxnext_(x,u)
    // compose by jacobians of log (xout - y)
    out_space().Jdifference(y, data_.xout_, data_.Jy_, 0);  // d(xnext - y) / y
    data_.Jtemp_.setZero();
    out_space().Jdifference(y, data_.xout_, data_.Jtemp_, 1);  // d(xnext - y) / xnext
    data_.Jx_ = data_.Jtemp_ * data_.Jx_;  // chain rule d(log)/dxnext * dxnext_dx
    data_.Ju_ = data_.Jtemp_ * data_.Ju_;
  }
  
  template<typename Scalar>
  shared_ptr<DynamicsDataTpl<Scalar>>
  ExplicitDynamicsModelTpl<Scalar>::createData() const
  {
    return std::make_shared<ExplicitData>(this->ndx1, this->nu, this->out_space());
  }
} // namespace proxddp
