#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {
namespace python {

namespace context {
using Scalar = double;

PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

using Manifold = ManifoldAbstractTpl<Scalar>;

using StageFunction = StageFunctionTpl<Scalar>;
using StageFunctionData = FunctionDataTpl<Scalar>;
using StageConstraint = StageConstraintTpl<Scalar>;

using ConstraintSet = ConstraintSetBase<Scalar>;

using CostBase = CostAbstractTpl<Scalar>;
using DynamicsModel = DynamicsModelTpl<Scalar>;
using StageModel = StageModelTpl<Scalar>;

using TrajOptProblem = TrajOptProblemTpl<Scalar>;
using TrajOptData = TrajOptDataTpl<Scalar>;

using ExplicitDynamics = ExplicitDynamicsModelTpl<Scalar>;
using ExplicitDynData = ExplicitDynamicsDataTpl<Scalar>;

} // namespace context

} // namespace python
} // namespace proxddp
