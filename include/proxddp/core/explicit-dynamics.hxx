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
    auto d = static_cast<SpecificData&>(data);
    this->forward(x, u, d.xout_);
    out_space_.difference(y, d.xout_, d.value_);  // xnext - y
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
    MatrixXs Jtemp(this->ndx2, this->ndx2);
    out_space_.Jdifference(y, d.xout_, d.Jy_, 0);  // d(xnext - y) / y
    Jtemp.setZero();
    out_space_.Jdifference(y, d.xout_, Jtemp, 1);  // d(xnext - y) / xnext
    d.Jx_ = (Jtemp * d.Jx_).eval();  // chain rule d(log)/dxnext * dxnext_dx
    d.Ju_ = (Jtemp * d.Ju_).eval();
  }
  
} // namespace proxddp
