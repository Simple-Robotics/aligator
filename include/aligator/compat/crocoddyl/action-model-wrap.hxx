/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/compat/crocoddyl/action-model-wrap.hpp"
#include "aligator/compat/crocoddyl/cost-wrap.hpp"
#include "aligator/core/vector-space.hpp"

namespace aligator {
namespace compat {
namespace croc {

template <typename Scalar>
ActionModelWrapperTpl<Scalar>::ActionModelWrapperTpl(
    shared_ptr<CrocActionModel> action_model)
    : Base(CostAbstract{StateWrapper{action_model->get_state()},
                        int(action_model->get_nu())},
           NoOpDynamics<Scalar>{StateWrapper{action_model->get_state()},
                                int(action_model->get_nu())})
    , action_model_(action_model) {}

template <typename Scalar>
void ActionModelWrapperTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                             const ConstVectorRef &u,
                                             Data &data) const {
  ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
  CrocActionModel &m = *action_model_;
  m.calc(d.croc_action_data, x, u);

  ALIGATOR_NOMALLOC_SCOPED;
  d.cost_data->value_ = d.croc_action_data->cost;
  DynDataWrap &dyn_data = *d.dynamics_data;
  dyn_data.xnext_ = d.croc_action_data->xnext;
}

template <typename Scalar>
void ActionModelWrapperTpl<Scalar>::computeFirstOrderDerivatives(
    const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const {
  ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
  CrocActionModel &m = *action_model_;
  m.calcDiff(d.croc_action_data, x, u);

  ALIGATOR_NOMALLOC_SCOPED;
  d.cost_data->Lx_ = d.croc_action_data->Lx;
  d.cost_data->Lu_ = d.croc_action_data->Lu;
  d.cost_data->Lxx_ = d.croc_action_data->Lxx;
  d.cost_data->Lxu_ = d.croc_action_data->Lxu;
  d.cost_data->Luu_ = d.croc_action_data->Luu;

  /* handle dynamics */
  DynDataWrap &dyn_data = *d.dynamics_data;
  dyn_data.Jx() = d.croc_action_data->Fx;
  dyn_data.Ju() = d.croc_action_data->Fu;
}

template <typename Scalar>
auto ActionModelWrapperTpl<Scalar>::createData() const -> shared_ptr<Data> {
  return std::make_shared<ActionDataWrap>(*this);
}

/* CrocActionDataWrapper */

template <typename Scalar>
ActionDataWrapperTpl<Scalar>::ActionDataWrapperTpl(
    const ActionModelWrapperTpl<Scalar> &action_model_wrap)
    : Base(action_model_wrap)
    , croc_action_data(action_model_wrap.action_model_->createData()) {
  dynamics_data =
      std::make_shared<DynamicsDataWrapper>(*action_model_wrap.action_model_);
  Base::dynamics_data = dynamics_data;
  this->cost_data =
      std::make_shared<CrocCostDataWrapperTpl<Scalar>>(croc_action_data);
  checkData();
}

template <typename Scalar> void ActionDataWrapperTpl<Scalar>::checkData() {
  if (croc_action_data == 0)
    ALIGATOR_RUNTIME_ERROR(
        "[StageData] integrity check failed: Crocoddyl action data is NULL.");
  Base::checkData();
}

} // namespace croc
} // namespace compat
} // namespace aligator
