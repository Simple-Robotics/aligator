#pragma once

#include "aligator/core/cost-abstract.hpp"
// Faster than std::unordered_map with Bost 1.80
// https://martin.ankerl.com/2022/08/27/hashmap-bench-01/#boost__unordered_map
#include <boost/unordered_map.hpp>

namespace aligator {

template <typename Scalar> struct CostStackDataTpl;

/** @brief Weighted sum of multiple cost components.
 *
 * @details This is expressed as
 * \f[
 *    \ell(x, u) = \sum_{k=1}^{K} \ell^{(k)}(x, u).
 * \f]
 */
template <typename _Scalar> struct CostStackTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using CostBase = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using PolyCost = xyz::polymorphic<CostBase>;
  using SumCostData = CostStackDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostItem = std::pair<PolyCost, Scalar>;
  using CostKey = std::variant<std::size_t, std::string>;
  using CostMap = boost::unordered::unordered_map<CostKey, CostItem>;
  using CostIterator = typename CostMap::iterator;

  CostMap components_;

  /// @brief    Check the dimension of a component.
  /// @returns  A bool value indicating whether the component is OK to be added
  /// to this instance.
  bool checkDimension(const CostBase &comp) const;

  /// @brief  Constructor with a specified dimension, and optional vector of
  /// components and weights.
  CostStackTpl(xyz::polymorphic<Manifold> space, const int nu,
               const std::vector<PolyCost> &comps = {},
               const std::vector<Scalar> &weights = {});

  CostStackTpl(xyz::polymorphic<Manifold> space, const int nu,
               const CostMap &comps)
      : CostBase(space, nu), components_(comps) {
    for (const auto &[key, item] : comps) {
      auto &cost = *item.first;
      if (!this->checkDimension(cost)) {
        ALIGATOR_DOMAIN_ERROR(fmt::format(
            "Cannot add new component due to inconsistent input dimensions "
            "(got ({:d}, {:d}), expected ({:d}, {:d}))",
            cost.ndx(), cost.nu, this->ndx(), this->nu));
      }
    }
  }

  /// @brief  Constructor from a single CostBase instance.
  CostStackTpl(const PolyCost &cost);

  inline CostItem &addCost(const PolyCost &cost, const Scalar weight = 1.) {
    const std::size_t size = components_.size();
    return this->addCost(size, cost, weight);
  }

  CostItem &addCost(const CostKey &key, const PolyCost &cost,
                    const Scalar weight = 1.);

  inline std::size_t size() const { return components_.size(); }

  /// @brief Get component, cast down to the specified type.
  template <typename Derived> Derived *getComponent(const CostKey &key) {
    CostItem &item = components_.at(key);
    return dynamic_cast<Derived *>(&*item.first);
  }

  template <typename Derived>
  const Derived *getComponent(const CostKey &key) const {
    CostItem &item = components_.at(key);
    return dynamic_cast<const Derived *>(&*item.first);
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const;

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data) const;

  shared_ptr<CostData> createData() const;
};

namespace {
template <typename T>
using CostPtr =
    xyz::polymorphic<CostAbstractTpl<T>>; //< convenience typedef for rest
                                          // of this file
}

template <typename T>
xyz::polymorphic<CostStackTpl<T>> operator+(const CostPtr<T> &c1,
                                            const CostPtr<T> &c2) {
  return xyz::polymorphic<CostStackTpl<T>>({c1, c2}, {1., 1.});
}

template <typename T>
xyz::polymorphic<CostStackTpl<T>>
operator+(xyz::polymorphic<CostStackTpl<T>> &&c1, const CostPtr<T> &c2) {
  c1->addCost(c2, 1.);
  return c1;
}

template <typename T>
xyz::polymorphic<CostStackTpl<T>>
operator+(xyz::polymorphic<CostStackTpl<T>> &&c1, CostPtr<T> &&c2) {
  c1->addCost(std::move(c2), 1.);
  return c1;
}

template <typename T>
xyz::polymorphic<CostStackTpl<T>>
operator+(const xyz::polymorphic<CostStackTpl<T>> &c1, CostPtr<T> &&c2) {
  c1->addCost(std::move(c2), 1.);
  return c1;
}

template <typename T>
xyz::polymorphic<CostStackTpl<T>>
operator*(T u, xyz::polymorphic<CostStackTpl<T>> &&c1) {
  for (auto &[key, item] : c1->components_) {
    item.second *= u;
  }
  return c1;
}

template <typename _Scalar>
struct CostStackDataTpl : CostDataAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  using CostData = CostDataAbstractTpl<Scalar>;
  using CostStack = CostStackTpl<Scalar>;
  using CostKey = typename CostStack::CostKey;
  using DataMap =
      boost::unordered::unordered_map<CostKey, shared_ptr<CostData>>;
  DataMap sub_cost_data;
  CostStackDataTpl(const CostStackTpl<Scalar> &obj);
};
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/costs/sum-of-costs.txx"
#endif
