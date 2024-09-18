/// @file  context.hpp
/// @brief Defines the context for instantiating the templates.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once
#include "aligator/compat/crocoddyl/fwd.hpp"
#include "aligator/context.hpp"

namespace aligator {
namespace compat {
namespace croc {

namespace context {

using Scalar = ::aligator::context::Scalar;

using StateWrapper = StateWrapperTpl<Scalar>;
using CostModelWrapper = CrocCostModelWrapperTpl<Scalar>;
using CostDataWrapper = CrocCostDataWrapperTpl<Scalar>;
using DynamicsDataWrapper = DynamicsDataWrapperTpl<Scalar>;
using ActionModelWrapper = ActionModelWrapperTpl<Scalar>;
using ActionDataWrapper = ActionDataWrapperTpl<Scalar>;

using CrocCostModel = crocoddyl::CostModelAbstractTpl<Scalar>;
using CrocCostData = crocoddyl::CostDataAbstractTpl<Scalar>;
using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
using CrocActionData = crocoddyl::ActionDataAbstractTpl<Scalar>;
using CrocShootingProblem = crocoddyl::ShootingProblemTpl<Scalar>;

} // namespace context
} // namespace croc

} // namespace compat

} // namespace aligator
