#pragma once

#include "proxddp/core/cost-abstract.hpp"
#include <crocoddyl/core/cost-base.hpp>

namespace proxddp {
namespace compat {
namespace croc {

template <typename _Scalar>
struct CrocCostWrapperTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  using CrocCostModel = ::crocoddyl::CostModelAbstractTpl<Scalar>;
  using CrocCostData = ::crocoddyl::CostDataAbstractTpl<Scalar>;
  boost::shared_ptr<CrocCostModel> croc_cost_;

  using BaseData = CostDataAbstractTpl<Scalar>;

  CrocCostWrapperTpl(boost::shared_ptr<CrocCostModel> cost)
      : croc_cost_(cost) {}
};

template <typename Scalar>
struct CrocCostDataWrapper : CostDataAbstractTpl<Scalar> {
  using CrocCostData = ::crocoddyl::CostDataAbstractTpl<Scalar>;
  boost::shared_ptr<CrocCostData> croc_data_;
};

} // namespace croc
} // namespace compat
} // namespace proxddp
