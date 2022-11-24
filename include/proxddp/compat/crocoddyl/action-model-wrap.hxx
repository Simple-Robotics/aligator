/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/compat/crocoddyl/action-model-wrap.hpp"

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar>
CrocActionModelWrapperTpl<Scalar>::CrocActionModelWrapperTpl(
    boost::shared_ptr<CrocActionModel> action_model)
    : Base(std::make_shared<StateWrapper>(action_model->get_state()),
           (int)action_model->get_nu()),
      action_model_(action_model) {
  using EqualitySet = proxnlp::EqualityConstraint<Scalar>;
  const int nr = (int)action_model->get_state()->get_ndx();
  this->constraints_.push_back(
      ConstraintType{nullptr, std::make_shared<EqualitySet>()}, nr);
}

template <typename Scalar>
void CrocActionModelWrapperTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                 const ConstVectorRef &u,
                                                 const ConstVectorRef &y,
                                                 Data &data) const {
  using dyn_data_t = DynamicsDataWrapperTpl<Scalar>;
  ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
  CrocActionModel &m = *action_model_;
  m.calc(d.croc_data, x, u);

  PROXDDP_NOMALLOC_BEGIN;
  d.cost_data->value_ = d.croc_data->cost;
  dyn_data_t &dyn_data = static_cast<dyn_data_t &>(*d.constraint_data[0]);
  dyn_data.xnext_ = d.croc_data->xnext;
  this->xspace_next_->difference(y, dyn_data.xnext_, dyn_data.value_);
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void CrocActionModelWrapperTpl<Scalar>::computeDerivatives(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    Data &data) const {
  using dyn_data_t = DynamicsDataWrapperTpl<Scalar>;
  ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
  CrocActionModel &m = *action_model_;
  m.calcDiff(d.croc_data, x, u);

  PROXDDP_NOMALLOC_BEGIN;
  d.cost_data->Lx_ = d.croc_data->Lx;
  d.cost_data->Lu_ = d.croc_data->Lu;
  d.cost_data->Lxx_ = d.croc_data->Lxx;
  d.cost_data->Lxu_ = d.croc_data->Lxu;
  d.cost_data->Luu_ = d.croc_data->Luu;

  /* handle dynamics */
  dyn_data_t &dyn_data = static_cast<dyn_data_t &>(*d.constraint_data[0]);
  dyn_data.Jx_ = d.croc_data->Fx;
  dyn_data.Ju_ = d.croc_data->Fu;
  this->xspace_next_->Jdifference(y, dyn_data.xnext_, dyn_data.Jy_, 0);
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
CrocActionDataWrapperTpl<Scalar>::CrocActionDataWrapperTpl(
    const CrocActionModel *croc_action_model,
    const boost::shared_ptr<CrocActionData> &action_data)
    : Base(), croc_data(action_data) {
  this->constraint_data = {
      std::make_shared<DynamicsDataWrapperTpl<Scalar>>(croc_action_model)};
  this->cost_data = std::make_shared<CrocCostDataWrapperTpl<Scalar>>(croc_data);
  checkData();
}

template <typename Scalar> void CrocActionDataWrapperTpl<Scalar>::checkData() {
  Base::checkData();
  if (croc_data == 0)
    PROXDDP_RUNTIME_ERROR("[StageData] integrity check failed.");
}

} // namespace croc
} // namespace compat
} // namespace proxddp
