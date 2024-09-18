#pragma once

#include "aligator/modelling/costs/sum-of-costs.hpp"

namespace aligator {
template <typename Scalar>
CostStackTpl<Scalar>::CostStackTpl(xyz::polymorphic<Manifold> space,
                                   const int nu,
                                   const std::vector<PolyCost> &comps,
                                   const std::vector<Scalar> &weights)
    : CostBase(space, nu) {
  if (comps.size() != weights.size()) {
    auto msg = fmt::format(
        "Inconsistent number of components ({:d}) and weights ({:d}).",
        comps.size(), weights.size());
    ALIGATOR_RUNTIME_ERROR(msg);
  } else {
    for (std::size_t i = 0; i < comps.size(); i++) {
      if (!this->checkDimension(*comps[i])) {
        auto msg = fmt::format("Component #{:d} has wrong input dimensions "
                               "({:d}, {:d}) (expected "
                               "({:d}, {:d}))",
                               i, comps[i]->ndx(), comps[i]->nu, this->ndx(),
                               this->nu);
        ALIGATOR_RUNTIME_ERROR(msg);
      }
    }

    for (std::size_t i = 0; i < comps.size(); i++) {
      components_.emplace(i, std::make_pair(comps[i], weights[i]));
    }
  }
}

template <typename Scalar>
CostStackTpl<Scalar>::CostStackTpl(const PolyCost &cost)
    : CostBase(cost->space, cost->nu) {
  components_.emplace(0UL, std::make_pair(cost, 1.0));
}

template <typename Scalar>
bool CostStackTpl<Scalar>::checkDimension(const CostBase &comp) const {
  return (comp.nx() == this->nx()) && (comp.ndx() == this->ndx()) &&
         (comp.nu == this->nu);
}

template <typename Scalar>
auto CostStackTpl<Scalar>::addCost(const CostKey &key, const PolyCost &cost,
                                   const Scalar weight) -> CostItem & {
  if (!this->checkDimension(*cost)) {
    ALIGATOR_DOMAIN_ERROR(fmt::format(
        "Cannot add new component due to inconsistent input dimensions "
        "(got ({:d}, {:d}), expected ({:d}, {:d}))",
        cost->ndx(), cost->nu, this->ndx(), this->nu));
  }
  components_.emplace(key, std::make_pair(cost, weight));
  return components_.at(key);
}

template <typename Scalar>
void CostStackTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    CostData &data) const {
  SumCostData &d = static_cast<SumCostData &>(data);
  d.value_ = 0.;
  for (const auto &[key, item] : components_) {
    item.first->evaluate(x, u, *d.sub_cost_data[key]);
    d.value_ += item.second * d.sub_cost_data[key]->value_;
  }
}

template <typename Scalar>
void CostStackTpl<Scalar>::computeGradients(const ConstVectorRef &x,
                                            const ConstVectorRef &u,
                                            CostData &data) const {
  SumCostData &d = static_cast<SumCostData &>(data);
  d.grad_.setZero();
  for (const auto &[key, item] : components_) {
    item.first->computeGradients(x, u, *d.sub_cost_data[key]);
    d.grad_.noalias() += item.second * d.sub_cost_data[key]->grad_;
  }
}

template <typename Scalar>
void CostStackTpl<Scalar>::computeHessians(const ConstVectorRef &x,
                                           const ConstVectorRef &u,
                                           CostData &data) const {
  SumCostData &d = static_cast<SumCostData &>(data);
  d.hess_.setZero();
  for (const auto &[key, item] : components_) {
    item.first->computeHessians(x, u, *d.sub_cost_data[key]);
    d.hess_.noalias() += item.second * d.sub_cost_data[key]->hess_;
  }
}

template <typename Scalar>
shared_ptr<CostDataAbstractTpl<Scalar>>
CostStackTpl<Scalar>::createData() const {
  return std::make_shared<SumCostData>(*this);
}

/* SumCostData */

template <typename Scalar>
CostStackDataTpl<Scalar>::CostStackDataTpl(const CostStackTpl<Scalar> &obj)
    : CostData(obj.ndx(), obj.nu) {
  for (const auto &[key, item] : obj.components_) {
    sub_cost_data[key] = item.first->createData();
  }
}

} // namespace aligator
