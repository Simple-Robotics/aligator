/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {
namespace context {

using Scalar = double;
static constexpr int Options = 0;

ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

using Manifold = ManifoldAbstractTpl<Scalar>;

using BCLParams = BCLParamsTpl<Scalar>;
using StageFunction = StageFunctionTpl<Scalar>;
using UnaryFunction = UnaryFunctionTpl<Scalar>;
using StageFunctionData = StageFunctionDataTpl<Scalar>;
using StageConstraint = StageConstraintTpl<Scalar>;
using CommonModel = CommonModelTpl<Scalar>;
using CommonModelData = CommonModelDataTpl<Scalar>;
using CommonModelBuilder = CommonModelBuilderTpl<Scalar>;
using CommonModelContainer = CommonModelContainerTpl<Scalar>;
using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;
using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;

using ConstraintSet = ConstraintSetBase<Scalar>;

using CostBase ALIGATOR_DEPRECATED_MESSAGE(
    "Use the CostAbstract typedef instead.") = CostAbstractTpl<Scalar>;
using CostAbstract = CostAbstractTpl<Scalar>;
using CostData = CostDataAbstractTpl<Scalar>;
using DynamicsModel = DynamicsModelTpl<Scalar>;
using StageModel = StageModelTpl<Scalar>;
using StageData = StageDataTpl<Scalar>;

using CallbackBase = CallbackBaseTpl<Scalar>;

using TrajOptProblem = TrajOptProblemTpl<Scalar>;
using TrajOptData = TrajOptDataTpl<Scalar>;
using ConstraintStack = ConstraintStackTpl<Scalar>;

using ExplicitDynamics = ExplicitDynamicsModelTpl<Scalar>;
using ExplicitDynamicsData = ExplicitDynamicsDataTpl<Scalar>;

using SolverProxDDP = SolverProxDDPTpl<Scalar>;
using SolverFDDP = SolverFDDPTpl<Scalar>;

using Workspace = WorkspaceTpl<Scalar>;
using Results = ResultsTpl<Scalar>;
using Filter = FilterTpl<Scalar>;

} // namespace context
} // namespace aligator
