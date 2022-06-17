#pragma once

#include "proxddp/core/costs.hpp"
#include <vector>


namespace proxddp
{

  /** @brief Weighted sum of multiple cost components.
   *
   * @details This is expressed as
   * \f[
   *    \ell(x, u) = \sum_{k=1}^{K} \ell^{(k)}(x, u).
   * \f]
   */
  template<typename _Scalar>
  struct CostStack
    : CostAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Base = CostAbstractTpl<Scalar>;
    using CostData = CostDataAbstract<Scalar>;
    using Self = CostStack<Scalar>;
    using VectorOfCosts = std::vector<shared_ptr<Base>>;

    /// Specific data holding struct for CostStack.
    struct SumCostData : CostData
    {
      std::vector<shared_ptr<CostData>> sub_datas;
      using CostData::CostDataAbstract;
    };

    VectorOfCosts components_;
    std::vector<Scalar> weights_;

    CostStack(const int ndx, const int nu, const VectorOfCosts& comps, const std::vector<Scalar>& weights)
      : Base(ndx, nu)
      , components_(comps)
      , weights_(weights)
    {
      assert(comps.size() == weights.size());
    }

    CostStack(const shared_ptr<Base>& comp)
      : CostStack(comp->ndx(), comp->nu(), {comp}, {1.})
      {}

    void addCost(const shared_ptr<Base>& cost, const Scalar weight = 1.)
    {
      components_.push_back(cost);
      weights_.push_back(weight);
    }

    std::size_t size() const
    {
      return components_.size();
    }

    void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {
      SumCostData& d = static_cast<SumCostData&>(data);
      d.value_ = 0.;
      for (std::size_t i = 0; i < components_.size(); i++)
      {
        components_[i]->evaluate(x, u, *d.sub_datas[i]);
        d.value_ += d.sub_datas[i]->value_;
      }
    }

    void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {
      SumCostData& d = static_cast<SumCostData&>(data);
      for (std::size_t i = 0; i < components_.size(); i++)
      {
        components_[i]->computeGradients(x, u, *d.sub_datas[i]);
        d.grad_.noalias() += d.sub_datas[i]->grad_;
      }
    }

    void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {
      SumCostData& d = static_cast<SumCostData&>(data);
      for (std::size_t i = 0; i < components_.size(); i++)
      {
        components_[i]->computeHessians(x, u, *d.sub_datas[i]);
        d.hess_.noalias() += d.sub_datas[i]->hess_;
      }
    }
  };

  template<typename T>
  shared_ptr<CostStack<T>> operator+(const shared_ptr<CostAbstractTpl<T>>& c1, const shared_ptr<CostAbstractTpl<T>>& c2)
  {
    return std::make_shared<CostStack<T>>({c1, c2}, {1., 1.});
  }

  template<typename T>
  shared_ptr<CostStack<T>> operator+(shared_ptr<CostStack<T>>&& c1, const shared_ptr<CostAbstractTpl<T>>& c2)
  {
    c1->addCost(c2, 1.);
    return std::move(c1);
  }

  template<typename T>
  shared_ptr<CostStack<T>> operator+(shared_ptr<CostStack<T>>&& c1, shared_ptr<CostAbstractTpl<T>>&& c2)
  {
    c1->addCost(std::move(c2), 1.);
    return std::move(c1);
  }

  template<typename T>
  shared_ptr<CostStack<T>> operator+(const shared_ptr<CostStack<T>>& c1, shared_ptr<CostAbstractTpl<T>>&& c2)
  {
    c1->addCost(std::move(c2), 1.);
    return c1;
  }

  template<typename T>
  shared_ptr<CostStack<T>> operator*(T u, const shared_ptr<CostAbstractTpl<T>>& c1)
  {
    return std::make_shared<CostStack<T>>({c1}, {u});
  }

  template<typename T>
  shared_ptr<CostStack<T>> operator*(T u, shared_ptr<CostStack<T>>&& c1)
  {
    for (std::size_t i = 0; i < c1->size(); i++)
    {
      c1->weights_[i] *= u;
    }
    return std::move(c1);
  }

} // namespace proxddp

