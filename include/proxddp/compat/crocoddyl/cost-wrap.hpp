#pragma once

#include "proxddp/compat/crocoddyl/fwd.hpp"
#include "proxddp/core/cost-abstract.hpp"
#include <crocoddyl/core/cost-base.hpp>
#include <crocoddyl/core/action-base.hpp>

#include <stdexcept>

namespace proxddp {
namespace compat {
namespace croc {

template <typename _Scalar>
struct CrocCostModelWrapperTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using CrocCostModel = crocoddyl::CostModelAbstractTpl<Scalar>;
  using Base = CostAbstractTpl<Scalar>;
  using BaseData = CostDataAbstractTpl<Scalar>;

  boost::shared_ptr<CrocCostModel> croc_cost_;
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> action_model_;

  /// Constructor from a crocoddyl cost model.
  explicit CrocCostModelWrapperTpl(boost::shared_ptr<CrocCostModel> cost)
      : Base(cost->get_state()->get_ndx(), cost->get_nu()), croc_cost_(cost) {}

  /// Constructor using a terminal action model.
  explicit CrocCostModelWrapperTpl(
      boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> action_model)
      : Base(action_model->get_state()->get_ndx(), action_model->get_nu()),
        action_model_(action_model) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const {
    using Data = CrocCostDataWrapperTpl<Scalar>;
    Data &d = static_cast<Data &>(data);
    if (croc_cost_ != 0) {
      croc_cost_->calc(d.croc_cost_data_, x, u);
      d.value_ = d.croc_cost_data_->cost;
    } else {
      action_model_->calc(d.croc_act_data_, x);
      d.value_ = d.croc_act_data_->cost;
    }
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const {
    using Data = CrocCostDataWrapperTpl<Scalar>;
    Data &d = static_cast<Data &>(data);
    if (croc_cost_ != 0) {
      croc_cost_->calcDiff(d.croc_cost_data_, x, u);
      d.Lx_ = d.croc_cost_data_->Lx;
      d.Lu_ = d.croc_cost_data_->Lu;
    } else {
      action_model_->calcDiff(d.croc_act_data_, x);
      d.Lx_ = d.croc_act_data_->Lx;
      d.Lu_ = d.croc_act_data_->Lu;
    }
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const {
    using Data = CrocCostDataWrapperTpl<Scalar>;
    Data &d = static_cast<Data &>(data);
    if (croc_cost_ != 0) {
      croc_cost_->calcDiff(d.croc_cost_data_, x, u);
      d.Lxx_ = d.croc_cost_data_->Lxx;
      d.Lxu_ = d.croc_cost_data_->Lxu;
      d.Luu_ = d.croc_cost_data_->Luu;
    } else {
      action_model_->calcDiff(d.croc_act_data_, x);
      d.Lxx_ = d.croc_act_data_->Lxx;
      d.Lxu_ = d.croc_act_data_->Lxu;
      d.Luu_ = d.croc_act_data_->Luu;
    }
  }

  shared_ptr<BaseData> createData() const {
    if (action_model_ != 0) {
      boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> am_data =
          action_model_->createData();
      return std::make_shared<CrocCostDataWrapperTpl<Scalar>>(am_data);
    } else {
      throw std::domain_error("Invalid call. Cannot build Data from"
                              "crocoddyl cost model only.");
    }
  }
};

template <typename Scalar>
struct CrocCostDataWrapperTpl : CostDataAbstractTpl<Scalar> {
  using CostData = ::crocoddyl::CostDataAbstractTpl<Scalar>;
  using ActionData = ::crocoddyl::ActionDataAbstractTpl<Scalar>;
  using Base = CostDataAbstractTpl<Scalar>;
  boost::shared_ptr<CostData> croc_cost_data_;
  boost::shared_ptr<ActionData> croc_act_data_;

  explicit CrocCostDataWrapperTpl(const boost::shared_ptr<CostData> &crocdata)
      : Base(crocdata->Lx.rows(), crocdata->Lu.rows()),
        croc_cost_data_(crocdata) {}

  explicit CrocCostDataWrapperTpl(const boost::shared_ptr<ActionData> &actdata)
      : Base(actdata->Lx.rows(), actdata->Lu.rows()), croc_act_data_(actdata) {}
};

} // namespace croc
} // namespace compat
} // namespace proxddp
