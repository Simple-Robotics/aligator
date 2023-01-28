/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {
namespace context {

using Scalar = double;

PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

using Manifold = ManifoldAbstractTpl<Scalar>;

using StageFunction = StageFunctionTpl<Scalar>;
using FunctionData = FunctionDataTpl<Scalar>;
using StageConstraint = StageConstraintTpl<Scalar>;

using ConstraintSet = ConstraintSetBase<Scalar>;

using CostBase = CostAbstractTpl<Scalar>;
using CostData = CostDataAbstractTpl<Scalar>;
using DynamicsModel = DynamicsModelTpl<Scalar>;
using StageModel = StageModelTpl<Scalar>;
using StageData = StageDataTpl<Scalar>;

using TrajOptProblem = TrajOptProblemTpl<Scalar>;
using TrajOptData = TrajOptDataTpl<Scalar>;

using ExplicitDynamics = ExplicitDynamicsModelTpl<Scalar>;
using ExplicitDynData = ExplicitDynamicsDataTpl<Scalar>;

using SolverProxDDP = ::proxddp::SolverProxDDP<Scalar>;
using SolverFDDP = ::proxddp::SolverFDDP<Scalar>;

using Workspace = WorkspaceTpl<Scalar>;
using Results = ResultsTpl<Scalar>;

} // namespace context
} // namespace proxddp
