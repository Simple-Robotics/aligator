/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {
namespace context {

using Scalar = double;
static constexpr int Options = 0;

ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

using Manifold = ManifoldAbstractTpl<Scalar>;
using VectorSpace = VectorSpaceTpl<Scalar, Eigen::Dynamic>;

using StageFunction = StageFunctionTpl<Scalar>;
using UnaryFunction = UnaryFunctionTpl<Scalar>;
using StageFunctionData = StageFunctionDataTpl<Scalar>;
using StageConstraint ALIGATOR_DEPRECATED = StageConstraintTpl<Scalar>;

using ConstraintSet = ConstraintSetTpl<Scalar>;

using CostBase ALIGATOR_DEPRECATED_MESSAGE(
    "Use the CostAbstract typedef instead.") = CostAbstractTpl<Scalar>;
using CostAbstract = CostAbstractTpl<Scalar>;
using CostData = CostDataAbstractTpl<Scalar>;
using DynamicsModel = DynamicsModelTpl<Scalar>;
using DynamicsData = DynamicsDataTpl<Scalar>;
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
