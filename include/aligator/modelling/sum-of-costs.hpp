#pragma once

#include "aligator/core/cost-abstract.hpp"

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
  using CostPtr = shared_ptr<CostBase>;
  using SumCostData = CostStackDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  std::vector<CostPtr> components_;
  std::vector<Scalar> weights_;

  /// @brief    Check the dimension of a component.
  /// @returns  A bool value indicating whether the component is OK to be added
  /// to this instance.
  bool checkDimension(const CostBase *comp) const;

  /// @brief  Constructor with a specified dimension, and optional vector of
  /// components and weights.
  CostStackTpl(shared_ptr<Manifold> space, const int nu,
               const std::vector<CostPtr> &comps = {},
               const std::vector<Scalar> &weights = {});

  /// @brief  Constructor from a single CostBase instance.
  CostStackTpl(const CostPtr &cost);

  void addCost(const CostPtr &cost, const Scalar weight = 1.);

  std::size_t size() const;

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
using CostPtr = shared_ptr<CostAbstractTpl<T>>; //< convenience typedef for rest
                                                // of this file
}

template <typename T>
shared_ptr<CostStackTpl<T>> operator+(const CostPtr<T> &c1,
                                      const CostPtr<T> &c2) {
  return std::make_shared<CostStackTpl<T>>({c1, c2}, {1., 1.});
}

template <typename T>
shared_ptr<CostStackTpl<T>> operator+(shared_ptr<CostStackTpl<T>> &&c1,
                                      const CostPtr<T> &c2) {
  c1->addCost(c2, 1.);
  return std::move(c1);
}

template <typename T>
shared_ptr<CostStackTpl<T>> operator+(shared_ptr<CostStackTpl<T>> &&c1,
                                      CostPtr<T> &&c2) {
  c1->addCost(std::move(c2), 1.);
  return std::move(c1);
}

template <typename T>
shared_ptr<CostStackTpl<T>> operator+(const shared_ptr<CostStackTpl<T>> &c1,
                                      CostPtr<T> &&c2) {
  c1->addCost(std::move(c2), 1.);
  return c1;
}

template <typename T>
shared_ptr<CostStackTpl<T>> operator*(T u, const CostPtr<T> &c1) {
  return std::make_shared<CostStackTpl<T>>({c1}, {u});
}

template <typename T>
shared_ptr<CostStackTpl<T>> operator*(T u, shared_ptr<CostStackTpl<T>> &&c1) {
  for (std::size_t i = 0; i < c1->size(); i++) {
    c1->weights_[i] *= u;
  }
  return std::move(c1);
}

template <typename _Scalar>
struct CostStackDataTpl : CostDataAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  using CostData = CostDataAbstractTpl<Scalar>;
  std::vector<shared_ptr<CostData>> sub_cost_data;
  CostStackDataTpl(const CostStackTpl<Scalar> &obj);
};
} // namespace aligator

#include "aligator/modelling/sum-of-costs.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/sum-of-costs.txx"
#endif
