#pragma once

#include "aligator/modelling/sum-of-costs.hpp"

namespace aligator {
template <typename Scalar>
CostStackTpl<Scalar>::CostStackTpl(shared_ptr<Manifold> space, const int nu,
                                   const std::vector<CostPtr> &comps,
                                   const std::vector<Scalar> &weights)
    : CostBase(space, nu), components_(comps), weights_(weights) {
  if (comps.size() != weights.size()) {
    auto msg = fmt::format(
        "Inconsistent number of components ({:d}) and weights ({:d}).",
        comps.size(), weights.size());
    ALIGATOR_RUNTIME_ERROR(msg);
  } else {
    for (std::size_t i = 0; i < comps.size(); i++) {
      if (!this->checkDimension(comps[i].get())) {
        auto msg = fmt::format(
            "Component #{:d} has wrong input dimensions ({:d}, {:d}) (expected "
            "({:d}, {:d}))",
            i, comps[i]->ndx(), comps[i]->nu, this->ndx(), this->nu);
        ALIGATOR_RUNTIME_ERROR(msg);
      }
    }
  }
}

template <typename Scalar>
CostStackTpl<Scalar>::CostStackTpl(const CostPtr &cost)
    : CostStackTpl(cost->space, cost->nu, {cost}, {1.}) {}

template <typename Scalar>
bool CostStackTpl<Scalar>::checkDimension(const CostBase *comp) const {
  return (comp->nx() == this->nx()) && (comp->ndx() == this->ndx()) &&
         (comp->nu == this->nu);
}

template <typename Scalar> std::size_t CostStackTpl<Scalar>::size() const {
  return components_.size();
}

template <typename Scalar>
void CostStackTpl<Scalar>::addCost(const CostPtr &cost, const Scalar weight) {
  if (!this->checkDimension(cost.get())) {
    ALIGATOR_DOMAIN_ERROR(fmt::format(
        "Cannot add new component due to inconsistent input dimensions "
        "(got ({:d}, {:d}), expected ({:d}, {:d}))",
        cost->ndx(), cost->nu, this->ndx(), this->nu));
  }
  components_.push_back(cost);
  weights_.push_back(weight);
}

template <typename Scalar>
void CostStackTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    CostData &data) const {
  SumCostData &d = static_cast<SumCostData &>(data);
  d.value_ = 0.;
  for (std::size_t i = 0; i < components_.size(); i++) {
    components_[i]->evaluate(x, u, *d.sub_cost_data[i]);
    d.value_ += this->weights_[i] * d.sub_cost_data[i]->value_;
  }
}

template <typename Scalar>
void CostStackTpl<Scalar>::computeGradients(const ConstVectorRef &x,
                                            const ConstVectorRef &u,
                                            CostData &data) const {
  SumCostData &d = static_cast<SumCostData &>(data);
  d.grad_.setZero();
  for (std::size_t i = 0; i < components_.size(); i++) {
    components_[i]->computeGradients(x, u, *d.sub_cost_data[i]);
    d.grad_.noalias() += this->weights_[i] * d.sub_cost_data[i]->grad_;
  }
}

template <typename Scalar>
void CostStackTpl<Scalar>::computeHessians(const ConstVectorRef &x,
                                           const ConstVectorRef &u,
                                           CostData &data) const {
  SumCostData &d = static_cast<SumCostData &>(data);
  d.hess_.setZero();
  for (std::size_t i = 0; i < components_.size(); i++) {
    components_[i]->computeHessians(x, u, *d.sub_cost_data[i]);
    d.hess_.noalias() += this->weights_[i] * d.sub_cost_data[i]->hess_;
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
  for (std::size_t i = 0; i < obj.size(); i++) {
    sub_cost_data.push_back(obj.components_[i]->createData());
  }
}

} // namespace aligator
