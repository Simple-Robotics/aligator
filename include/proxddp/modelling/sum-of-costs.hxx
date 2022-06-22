#pragma once

#include "proxddp/modelling/sum-of-costs.hpp"

namespace proxddp
{
  template<typename Scalar>
  std::size_t CostStack<Scalar>::size() const
  {
    return components_.size();
  }

  template<typename Scalar>
  void CostStack<Scalar>::addCost(const shared_ptr<CostBase>& cost, const Scalar weight)
  {
    components_.push_back(cost);
    weights_.push_back(weight);
  }

  template<typename Scalar>
  void CostStack<Scalar>::evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
  {
    SumCostData& d = static_cast<SumCostData&>(data);
    d.value_ = 0.;
    for (std::size_t i = 0; i < components_.size(); i++)
    {
      components_[i]->evaluate(x, u, *d.sub_datas[i]);
      d.value_ += d.sub_datas[i]->value_;
    }
  }

  template<typename Scalar>
  void CostStack<Scalar>::computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
  {
    SumCostData& d = static_cast<SumCostData&>(data);
    d.grad_.setZero();
    for (std::size_t i = 0; i < components_.size(); i++)
    {
      components_[i]->computeGradients(x, u, *d.sub_datas[i]);
      d.grad_.noalias() += d.sub_datas[i]->grad_;
    }
  }

  template<typename Scalar>
  void CostStack<Scalar>::computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
  {
    SumCostData& d = static_cast<SumCostData&>(data);
    d.hess_.setZero();
    for (std::size_t i = 0; i < components_.size(); i++)
    {
      components_[i]->computeHessians(x, u, *d.sub_datas[i]);
      d.hess_.noalias() += d.sub_datas[i]->hess_;
    }
  }
  
  template<typename Scalar>
  shared_ptr< CostDataAbstractTpl<Scalar> > CostStack<Scalar>::createData() const
  {
    return std::make_shared<SumCostData>(*this);
  }
} // namespace proxddp

