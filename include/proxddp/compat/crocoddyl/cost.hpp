#pragma once

#include "proxddp/core/cost-abstract.hpp"
#include <crocoddyl/core/cost-base.hpp>

#include <stdexcept>

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar> struct CrocCostDataWrapper;

template <typename _Scalar>
struct CrocCostWrapperTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using CrocCostModel = ::crocoddyl::CostModelAbstractTpl<Scalar>;
  using CrocCostData = ::crocoddyl::CostDataAbstractTpl<Scalar>;
  boost::shared_ptr<CrocCostModel> croc_cost_;

  using BaseData = CostDataAbstractTpl<Scalar>;

  CrocCostWrapperTpl(boost::shared_ptr<CrocCostModel> cost)
      : croc_cost_(cost) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) {
    using Data = CrocCostDataWrapper<Scalar>;
    Data &d = static_cast<Data &>(data);
    croc_cost_->calc(d.croc_data_, x, u);
    d.value_ = d.croc_data_->cost;
    d.Lx_ = d.croc_data_->Lx;
    d.Lu_ = d.croc_data_->Lu;
    d.Lxx_ = d.croc_data_->Lxx;
    d.Lxu_ = d.croc_data_->Lxu;
    d.Lux_ = d.croc_data_->Lux;
    d.Luu_ = d.croc_data_->Luu;
  }

  shared_ptr<BaseData> createData() const {
    throw std::domain_error("Invalid call. Construction of this class' Data "
                            "struct is handled elsewhere.");
  }
};

template <typename Scalar>
struct CrocCostDataWrapper : CostDataAbstractTpl<Scalar> {
  using CrocCostData = ::crocoddyl::CostDataAbstractTpl<Scalar>;
  using Base = CostDataAbstractTpl<Scalar>;
  boost::shared_ptr<CrocCostData> croc_data_;
  CrocCostDataWrapper(boost::shared_ptr<CrocCostData> crocdata)
      : Base(crocdata->Lx.rows(), crocdata->Lu.cols()), croc_data_(crocdata) {}
};

} // namespace croc
} // namespace compat
} // namespace proxddp
