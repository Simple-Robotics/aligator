#pragma once

#include "proxddp/core/costs.hpp"


namespace proxddp
{

  /// @brief Constant cost.
  template<typename _Scalar>
  struct ConstantCost : CostAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using CostData = CostDataAbstractTpl<Scalar>;

    Scalar value_;
    ConstantCost(const Scalar value) : value_(value) {}
    void evaluate(const ConstVectorRef&, const ConstVectorRef&, CostData& data) const
    {
      data.value_ = value_;
    }

    void computeGradients(const ConstVectorRef&, const ConstVectorRef&, CostData& data) const
    {
      data.grad_.setZero();
    }

    void computeHessians(const ConstVectorRef&, const ConstVectorRef&, CostData& data) const
    {
      data.hess_.setZero();
    }
  };


  /// @brief Euclidean quadratic cost.
  template<typename _Scalar>
  struct QuadraticCost : CostAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using CostData = CostDataAbstractTpl<Scalar>;

    MatrixXs weights_x;
    MatrixXs weights_u;
    VectorXs interp_x;
    VectorXs interp_u;

    QuadraticCost(const ConstMatrixRef& w_x,
                  const ConstMatrixRef& w_u,
                  const ConstVectorRef& interp_x,
                  const ConstVectorRef& interp_u)
      : CostAbstractTpl<_Scalar>((int)w_x.cols(), (int)w_u.cols())
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
