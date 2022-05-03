#pragma once

#include "proxddp/fwd.hpp"


namespace proxddp
{
  namespace python
  {

    namespace context
    {
      using Scalar = double;

      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

      using Manifold = ManifoldAbstractTpl<Scalar>;
      using Constraint = StageConstraintTpl<Scalar>;

      using StageFunction = StageFunctionTpl<Scalar>;
      using FunctionData = FunctionDataTpl<Scalar>;

      using CostBase = CostBaseTpl<Scalar>;

      using DynamicsModel = DynamicsModelTpl<Scalar>;      

      using ShootingProblem = ShootingProblemTpl<Scalar>;

    } // namespace context

  } // namespace python
} // namespace proxddp

