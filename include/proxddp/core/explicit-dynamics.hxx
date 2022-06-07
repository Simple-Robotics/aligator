#pragma once

#include "proxddp/core/explicit-dynamics.hpp"


namespace proxddp
{
  template<typename Scalar>
  void ExplicitDynamicsModelTpl<Scalar>::
  evaluate(const ConstVectorRef& x,
           const ConstVectorRef& u,
           const ConstVectorRef& y,
           Data& data) const
  {
    // Call the forward dynamics and set the function residual
    // value to the difference between y and the xout_.
    SpecificData& d = static_cast<SpecificData&>(data);
    this->forward(x, u, d.xout_);
    out_space().difference(y, d.xout_, d.value_);  // xnext - y
  }

  template<typename Scalar>
  void ExplicitDynamicsModelTpl<Scalar>::
  computeJacobians(const ConstVectorRef& x,
                   const ConstVectorRef& u,
                   const ConstVectorRef& y,
                   Data& data) const
  {
    SpecificData& d = static_cast<SpecificData&>(data);
    this->forward(x, u, d.xout_);
    this->dForward(x, u, d.Jx_, d.Ju_); // dxnext_(x,u)
    // compose by jacobians of log (xout - y)
    out_space().Jdifference(y, d.xout_, d.Jy_, 0);  // d(xnext - y) / y
    d.Jtemp_.setZero();
    out_space().Jdifference(y, d.xout_, d.Jtemp_, 1);  // d(xnext - y) / xnext
    d.Jx_ = d.Jtemp_ * d.Jx_;  // chain rule d(log)/dxnext * dxnext_dx
    d.Ju_ = d.Jtemp_ * d.Ju_;
  }
  
} // namespace proxddp
