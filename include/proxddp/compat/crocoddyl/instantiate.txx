/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/compat/crocoddyl/cost-wrap.hpp"
#include "proxddp/compat/crocoddyl/action-model-wrap.hpp"
#include "proxddp/compat/crocoddyl/problem-wrap.hpp"
#include "proxddp/compat/crocoddyl/context.hpp"

namespace proxddp {
namespace compat {
namespace croc {

extern template
::proxddp::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(const boost::shared_ptr<context::CrocShootingProblem> &);

extern template CrocCostModelWrapperTpl<context::Scalar>::CrocCostModelWrapperTpl(
    boost::shared_ptr<context::CrocCostModel>);

extern template CrocCostModelWrapperTpl<context::Scalar>::CrocCostModelWrapperTpl(
    boost::shared_ptr<context::CrocActionModel>);

extern template DynamicsDataWrapperTpl<context::Scalar>::DynamicsDataWrapperTpl(const context::CrocActionModel *);

extern template CrocCostDataWrapperTpl<context::Scalar>::CrocCostDataWrapperTpl(const boost::shared_ptr<context::CrocCostData> &);

extern template CrocCostDataWrapperTpl<context::Scalar>::CrocCostDataWrapperTpl(const boost::shared_ptr<context::CrocActionData> &);

} // namespace croc
} // namespace compat
} // namespace proxddp
