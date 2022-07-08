#pragma once

#include "proxddp/core/costs.hpp"
#include <crocoddyl/core/cost-base.hpp>

namespace proxddp {
namespace compat {
namespace croc {

template <typename _Scalar>
struct CostAbstractWrapper : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  using CrocCostModel = ::crocoddyl::CostModelAbstractTpl<Scalar>;
  shared_ptr<CrocCostModel> croc_cost;
};

} // namespace croc
} // namespace compat
} // namespace proxddp
