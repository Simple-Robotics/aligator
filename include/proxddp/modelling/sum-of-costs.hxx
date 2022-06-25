#pragma once

#include "proxddp/modelling/sum-of-costs.hpp"

#include <stdexcept>
#include <fmt/format.h>

namespace proxddp
{
  template<typename Scalar>
  bool CostStackTpl<Scalar>::checkDimension(const CostBase* comp) const
  {
    return (comp->ndx() == this->ndx()) && (comp->nu() == this->nu());
  }

  template<typename Scalar>
  CostStackTpl<Scalar>::
  CostStackTpl(const int ndx, const int nu, const VectorOfCosts& comps, const std::vector<Scalar>& weights)
    : CostBase(ndx, nu)
    , components_(comps)
    , weights_(weights)
  {
    if (comps.size() != weights.size())
    {
      std::string msg = fmt::format("Inconsistent number of components ({:d}) and weights ({:d}).", comps.size(), weights.size());
      throw std::domain_error(msg);
    } else {
      for (std::size_t i = 0; i < comps.size(); i++)
      {
        if (!this->checkDimension(comps[i].get()))
        {
          std::string msg = fmt::format(
            "Component #{:d} has wrong input dimensions ({:d}, {:d}) (expected ({:d}, {:d}))",
            i, comps[i]->ndx(), comps[i]->nu(), this->ndx(), this->nu());
          throw std::domain_error(msg);
        }
      }
    }
  }

  template<typename Scalar>
  CostStackTpl<Scalar>::
  CostStackTpl(const shared_ptr<CostBase>& comp): CostStackTpl(comp->ndx(), comp->nu(), {comp}, {1.}) {}

  template<typename Scalar>
  std::size_t CostStackTpl<Scalar>::size() const
  {
    return components_.size();
  }

  template<typename Scalar>
  void CostStackTpl<Scalar>::addCost(const shared_ptr<CostBase>& cost, const Scalar weight)
  {
    if (!this->checkDimension(cost.get()))
    {
      throw std::domain_error(fmt::format(
        "Cannot add new component due to inconsistent input dimensions "
        "(got ({:d}, {:d}), expected ({:d}, {:d}))",
        cost->ndx(), cost->nu(), this->ndx(), this->nu()));
    }
    components_.push_back(cost);
    weights_.push_back(weight);
  }

  template<typename Scalar>
  void CostStackTpl<Scalar>::evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
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
  void CostStackTpl<Scalar>::computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
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
  void CostStackTpl<Scalar>::computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
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
  shared_ptr< CostDataAbstractTpl<Scalar> > CostStackTpl<Scalar>::createData() const
  {
    return std::make_shared<SumCostData>(*this);
  }
} // namespace proxddp

