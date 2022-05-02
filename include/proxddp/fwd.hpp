///   Forward declarations.
#pragma once

#include "proxnlp/fwd.hpp"

/// Main package namespace.
namespace proxddp
{

  // Use the shared_ptr used in proxnlp.
  using proxnlp::shared_ptr;

  using proxnlp::math_types;
  using proxnlp::ManifoldAbstractTpl;

  // Using the constraint set from proxnlp
  using proxnlp::ConstraintSetBase;

  // fwd StageFunctionTpl
  template<typename Scalar>
  struct StageFunctionTpl;

  // fwd FunctionDataTpl
  template<typename Scalar>
  struct FunctionDataTpl;

  // fwd StageCostTpl
  template<typename Scalar>
  struct StageCostTpl;

  // fwd DynamicsModelTpl
  template<typename Scalar>
  struct DynamicsModelTpl;

  // fwd StageConstraintTpl
  template<typename Scalar>
  struct StageConstraintTpl;


  /* Stage models */

  // fwd StageModelTpl
  template<typename Scalar>
  class StageModelTpl;


  template<typename Scalar>
  struct StageDataTpl;


  /* Shooting problem */

  // fwd ShootingProblemTpl
  template<typename _Scalar>
  struct ShootingProblemTpl;


  /// Math utilities
  namespace math
  {
    using namespace proxnlp::math;
  } // namespace math

} // namespace proxddp

