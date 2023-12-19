/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/compat/crocoddyl/action-model-wrap.hpp"
#include "proxddp/core/stage-data.hpp"

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar>
ActionModelWrapperTpl<Scalar>::ActionModelWrapperTpl(
    boost::shared_ptr<CrocActionModel> action_model)
    : Base(std::make_shared<StateWrapper>(action_model->get_state()),
           (int)action_model->get_nu()),
      action_model_(action_model) {
  using EqualitySet = proxsuite::nlp::EqualityConstraint<Scalar>;
  const int nr = (int)action_model->get_state()->get_ndx();
  this->constraints_.pushBack(
      Constraint{nullptr, std::make_shared<EqualitySet>()}, nr);
}

template <typename Scalar>
void ActionModelWrapperTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                             const ConstVectorRef &u,
                                             const ConstVectorRef &y,
                                             Data &data) const {
  ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
  CrocActionModel &m = *action_model_;
  m.calc(d.croc_action_data, x, u);

  PROXDDP_NOMALLOC_BEGIN;
  d.cost_data->value_ = d.croc_action_data->cost;
  DynDataWrap &dyn_data = *d.dynamics_data;
  dyn_data.xnext_ = d.croc_action_data->xnext;
  this->xspace_next_->difference(y, dyn_data.xnext_, dyn_data.value_);
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
void ActionModelWrapperTpl<Scalar>::computeDerivatives(const ConstVectorRef &x,
                                                       const ConstVectorRef &u,
                                                       const ConstVectorRef &y,
                                                       Data &data) const {
  ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
  CrocActionModel &m = *action_model_;
  m.calcDiff(d.croc_action_data, x, u);

  PROXDDP_NOMALLOC_BEGIN;
  d.cost_data->Lx_ = d.croc_action_data->Lx;
  d.cost_data->Lu_ = d.croc_action_data->Lu;
  d.cost_data->Lxx_ = d.croc_action_data->Lxx;
  d.cost_data->Lxu_ = d.croc_action_data->Lxu;
  d.cost_data->Luu_ = d.croc_action_data->Luu;

  /* handle dynamics */
  DynDataWrap &dyn_data = *d.dynamics_data;
  dyn_data.Jx_ = d.croc_action_data->Fx;
  dyn_data.Ju_ = d.croc_action_data->Fu;
  this->xspace_next_->Jdifference(y, dyn_data.xnext_, dyn_data.Jy_, 0);
  PROXDDP_NOMALLOC_END;
}

template <typename Scalar>
auto ActionModelWrapperTpl<Scalar>::createData() const -> shared_ptr<Data> {
  return std::make_shared<ActionDataWrap>(action_model_);
}

/* CrocActionDataWrapper */

template <typename Scalar>
ActionDataWrapperTpl<Scalar>::ActionDataWrapperTpl(
    const boost::shared_ptr<CrocActionModel> &croc_action_model)
    : Base(), croc_action_data(croc_action_model->createData()) {
  dynamics_data = std::make_shared<DynamicsDataWrapper>(*croc_action_model);
  Base::dynamics_data = dynamics_data;
  this->constraint_data = {dynamics_data};
  this->cost_data =
      std::make_shared<CrocCostDataWrapperTpl<Scalar>>(croc_action_data);
  checkData();
}

template <typename Scalar> void ActionDataWrapperTpl<Scalar>::checkData() {
  if (croc_action_data == 0)
    PROXDDP_RUNTIME_ERROR(
        "[StageData] integrity check failed: Crocoddyl action data is NULL.");
  Base::checkData();
}

} // namespace croc
} // namespace compat
} // namespace proxddp
