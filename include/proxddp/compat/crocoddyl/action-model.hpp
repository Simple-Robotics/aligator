#pragma once

#include "proxddp/compat/crocoddyl/cost-wrap.hpp"
#include "proxddp/compat/crocoddyl/state-wrap.hpp"
#include "proxddp/compat/crocoddyl/dynamics-wrap.hpp"

#include "proxddp/core/stage-model.hpp"
#include <crocoddyl/core/action-base.hpp>

#include "proxddp/utils/exceptions.hpp"

namespace proxddp {
namespace compat {
namespace croc {

/**
 * @brief Wraps a crocoddyl::ActionModelAbstract
 *
 * This data structure rewires an ActionModel into a StageModel object.
 */
template <typename Scalar>
struct CrocActionModelWrapperTpl : public StageModelTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageModelTpl<Scalar>;
  using Data = StageDataTpl<Scalar>;
  using ConstraintType = typename Base::Constraint;
  using Dynamics = typename Base::Dynamics;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using StateWrapper = StateWrapperTpl<Scalar>;
  using ActionDataWrap = CrocActionDataWrapperTpl<Scalar>;

  boost::shared_ptr<CrocActionModel> action_model_;

  explicit CrocActionModelWrapperTpl(
      boost::shared_ptr<CrocActionModel> action_model)
      : Base(std::make_shared<StateWrapper>(action_model->get_state()),
             action_model->get_nu()),
        action_model_(action_model) {
    using EqualitySet = proxnlp::EqualityConstraint<Scalar>;
    const int nr = action_model->get_state()->get_ndx();
    this->constraints_.push_back(
        ConstraintType{nullptr, std::make_shared<EqualitySet>()}, nr);
  }

  const Dynamics &dyn_model() const {
    proxddp_runtime_error("There is not dyn_model() for this class.")
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const {
    ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
    CrocActionModel &m = *action_model_;
    m.calc(d.croc_data, x, u);
    d.cost_data->value_ = d.croc_data->cost;

    using dyn_data_t = DynamicsDataWrapperTpl<Scalar>;
    dyn_data_t &dyn_data = static_cast<dyn_data_t &>(*d.constraint_data[0]);
    dyn_data.xnext_ = d.croc_data->xnext;
    this->xspace_next_->difference(y, dyn_data.xnext_, dyn_data.value_);
  }

  void computeDerivatives(const ConstVectorRef &x, const ConstVectorRef &u,
                          const ConstVectorRef &y, Data &data) const {
    ActionDataWrap &d = static_cast<ActionDataWrap &>(data);
    CrocActionModel &m = *action_model_;
    m.calcDiff(d.croc_data, x, u);
    d.cost_data->Lx_ = d.croc_data->Lx;
    d.cost_data->Lu_ = d.croc_data->Lu;
    d.cost_data->Lxx_ = d.croc_data->Lxx;
    d.cost_data->Lxu_ = d.croc_data->Lxu;
    d.cost_data->Luu_ = d.croc_data->Luu;

    /* handle dynamics */
    using dyn_data_t = DynamicsDataWrapperTpl<Scalar>;
    dyn_data_t &dyn_data = static_cast<dyn_data_t &>(*d.constraint_data[0]);
    dyn_data.Jx_ = d.croc_data->Fx;
    dyn_data.Ju_ = d.croc_data->Fu;
    this->xspace_next_->Jdifference(y, dyn_data.xnext_, dyn_data.Jy_, 0);
    this->xspace_next_->Jdifference(y, dyn_data.xnext_, dyn_data.Jtmp_xnext, 1);
    dyn_data.Jx_ = dyn_data.Jtmp_xnext * dyn_data.Jx_;
    dyn_data.Ju_ = dyn_data.Jtmp_xnext * dyn_data.Ju_;
  }

  shared_ptr<Data> createData() const {
    using CrocActionData = crocoddyl::ActionDataAbstractTpl<Scalar>;
    boost::shared_ptr<CrocActionData> cd = action_model_->createData();
    return std::make_shared<ActionDataWrap>(action_model_.get(), std::move(cd));
  }
};

/**
 * @brief A complicated child class to StageDataTpl which pipes Crocoddyl's data
 * to the right places.
 */
template <typename Scalar>
struct CrocActionDataWrapperTpl : public StageDataTpl<Scalar> {
  using Base = StageDataTpl<Scalar>;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using CrocActionData = crocoddyl::ActionDataAbstractTpl<Scalar>;
  using Base::constraint_data;
  using Base::cost_data;

  boost::shared_ptr<CrocActionData> croc_data;

  CrocActionDataWrapperTpl(const CrocActionModel *croc_action_model,
                           const boost::shared_ptr<CrocActionData> &action_data)
      : Base(), croc_data(action_data) {
    constraint_data = {
        std::make_shared<DynamicsDataWrapperTpl<Scalar>>(croc_action_model)};
    cost_data = std::make_shared<CrocCostDataWrapperTpl<Scalar>>(croc_data);
    checkData();
  }

  void checkData() {
    Base::checkData();
    if (croc_data == 0)
      std::domain_error("[StageData] integrity check failed.");
  }
};

} // namespace croc
} // namespace compat
} // namespace proxddp

#include "proxddp/compat/crocoddyl/action-model.hxx"
