#pragma once
#include "proxddp/core/problem.hpp"
#include "proxddp/core/explicit-dynamics.hpp"



using namespace proxddp;

/// @brief    Addition dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x + u \f$.
struct MyModel : ExplicitDynamicsModelTpl<double>
{
  MyModel(const ManifoldAbstractTpl<double>& space)
    : ExplicitDynamicsModelTpl<double>(space, space.ndx()) {}  
  void forward(const ConstVectorRef& x, const ConstVectorRef& u, VectorRef out) const
  {
    out_space_.integrate(x, u, out);
  }

  void dForward(const ConstVectorRef& x, const ConstVectorRef& u, MatrixRef Jx, MatrixRef Ju) const
  {
    out_space_.Jintegrate(x, u, Jx, 0);
    out_space_.Jintegrate(x, u, Ju, 1);
  }
};


struct MyCost : CostBaseTpl<double>
{
  using CostBaseTpl<double>::CostBaseTpl;
  void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
  {
    data.value_ = 0.;
  }

  void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
  {
    data.grad_.setZero();
  }

  void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
  {
    data.hess_.setZero();
  }
};

