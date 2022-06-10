#pragma once

#include "proxddp/core/function.hpp"

namespace proxddp
{
  /// Useful for box constraints.
  ///
  /// \f\[ c(u) = [u_{\min} - u, u - u_{\max}] \f\]
  template<typename Scalar>
  struct ControlBoxFunction : StageFunctionTpl<Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Data = typename StageFunctionTpl<Scalar>::Data;

    VectorXs umin_, umax_;

    ControlBoxFunction(const int ndx, const int nu, VectorXs umin, VectorXs umax)
      : StageFunctionTpl<Scalar>(ndx, nu, ndx, nu * 2)
      , umin_(umin)
      , umax_(umax)
      {}

    ControlBoxFunction(const int ndx, const int nu, Scalar umin, Scalar umax)
      : ControlBoxFunction(ndx, nu, VectorXs::Constant(nu, umin), VectorXs::Constant(nu, umax))
      {}

    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  Data& data) const
    {
      data.value_.head(this->nu) = umin_ - u;
      data.value_.tail(this->nu) = u - umax_;
    }

    void computeJacobians(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& y,
                          Data& data) const
    {
      data.jac_buffer_.setZero();
      data.Ju_.topRows(this->nu).diagonal().array() = -1.;
      data.Ju_.bottomRows(this->nu).diagonal().array() = 1.;
    }
  };
  
} // namespace proxddp

