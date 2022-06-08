#pragma once

#include "proxddp/core/costs.hpp"
#include <vector>


namespace proxddp
{

  /// @brief Weighted sum of multiple cost components.
  template<typename _Scalar>
  struct SumOfCosts
    : CostAbstractTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Base = CostAbstractTpl<Scalar>;
    using CostData = CostDataAbstract<Scalar>;
    using Self = SumOfCosts<Scalar>;
    using VectorOfCosts = std::vector<shared_ptr<Base>>;

    /// Specific data holding struct for SumOfCosts.
    struct SumCostData : CostData
    {
      std::vector<shared_ptr<CostData>> sub_datas;
      using CostData::CostAbstractTpl;
    };

    VectorOfCosts components_;
    std::vector<Scalar> weights_;

    SumOfCosts(const VectorOfCosts& comps, const std::vector<Scalar>& weights)
      : components_(comps)
      , weights_(weights) {
      assert(comps.size() == weights.size());
    }

    SumOfCosts(const shared_ptr<Base>& comp)
      : components_({comp})
      , weights_({1.}) {}

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
      auto& d = static_cast<SumCostData&>(data);
      d.value_ = 0.;
      for (std::size_t i = 0; i < components_.size(); i++)
      {
        components_[i]->evaluate(x, u, *d.sub_datas[i]);
        d.value_ += d.sub_datas[i]->value_;
      }
    }

    void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {

    }

    void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const
    {

    }
  };

  template<typename T>
  shared_ptr<SumOfCosts<T>> operator+(const shared_ptr<CostAbstractTpl<T>>& c1, const shared_ptr<CostAbstractTpl<T>>& c2)
  {
    return std::make_shared<SumOfCosts<T>>({c1, c2}, {1., 1.});
  }

  template<typename T>
  shared_ptr<SumOfCosts<T>> operator+(shared_ptr<SumOfCosts<T>>&& c1, const shared_ptr<CostAbstractTpl<T>>& c2)
  {
    c1->addCost(c2, 1.);
    return std::move(c1);
  }

  template<typename T>
  shared_ptr<SumOfCosts<T>> operator*(T u, const shared_ptr<CostAbstractTpl<T>>& c1)
  {
    return std::make_shared<SumOfCosts<T>>({c1}, {u});
  }

  template<typename T>
  shared_ptr<SumOfCosts<T>> operator*(T u, shared_ptr<SumOfCosts<T>>&& c1)
  {
    for (std::size_t i = 0; i < c1->size(); i++)
    {
      c1->weights_[i] *= u;
    }
    return std::move(c1);
  }

} // namespace proxddp

