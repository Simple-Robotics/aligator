/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/compat/crocoddyl/action-model-wrap.hpp"
#include "proxddp/compat/crocoddyl/context.hpp"
#include "proxddp/compat/crocoddyl/cost-wrap.hpp"
#include "proxddp/compat/crocoddyl/problem-wrap.hpp"

namespace proxddp {
namespace compat {
namespace croc {

extern template ::proxddp::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);

extern template struct StateWrapperTpl<context::Scalar>;

extern template struct CrocCostModelWrapperTpl<context::Scalar>;
extern template struct CrocCostDataWrapperTpl<context::Scalar>;

extern template struct DynamicsDataWrapperTpl<context::Scalar>;

extern template struct ActionModelWrapperTpl<context::Scalar>;
extern template struct ActionDataWrapperTpl<context::Scalar>;

} // namespace croc
} // namespace compat
} // namespace proxddp
