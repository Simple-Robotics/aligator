#pragma once

#include "proxddp/core/costs.hpp"

namespace proxddp
{
  template<typename _Scalar>
  struct QuadraticCost : CostBaseTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using CostData = CostDataTpl<Scalar>;

    MatrixXs weights_x;
    MatrixXs weights_u;
    VectorXs interp_x;
    VectorXs interp_u;

    QuadraticCost(const ConstMatrixRef& w_x,
                  const ConstMatrixRef& w_u,
                  const ConstMatrixRef& interp_x,
                  const ConstMatrixRef& interp_u)
      : CostBaseTpl<_Scalar>((int)w_x.cols(), (int)w_u.cols())
      , weights_x(w_x)
      , weights_u(w_u)
      , interp_x(interp_x)
      , interp_u(interp_u)
      {}

    QuadraticCost(const ConstMatrixRef& w_x, const ConstMatrixRef& w_u)
      : QuadraticCost(w_x, w_u, VectorXs::Zero(w_x.cols()), VectorXs::Zero(w_u.cols())) {}

    void evaluate(
      const ConstVectorRef& x,
      const ConstVectorRef& u,
      CostData& data) const
    {
      data.value_ = \
        0.5 * x.dot(weights_x * x + 2 * interp_x) + \
        0.5 * u.dot(weights_u * u + 2 * interp_u);
    }

    void computeGradients(
      const ConstVectorRef& x,
      const ConstVectorRef& u,
      CostData& data) const
    {
      data.Lx_ = weights_x * x + interp_x;
      data.Lu_ = weights_u * u + interp_u;
    }

    void computeHessians(
      const ConstVectorRef&,
      const ConstVectorRef&,
      CostData& data) const
    {
      data.Lxx_ = weights_x;
      data.Luu_ = weights_u;
    }
  };
  
} // namespace proxddp
