#pragma once
/// @file fwd.hpp
/// @brief Forward declarations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxnlp/fwd.hpp"

/// Main package namespace.
namespace proxddp
{
  /// TYPEDEFS FROM PROXNLP

  // Use the shared_ptr used in proxnlp.
  using proxnlp::shared_ptr;
  using proxnlp::math_types;
  using proxnlp::ManifoldAbstractTpl;
  // Using the constraint set from proxnlp
  using proxnlp::ConstraintSetBase;


  /// 1 BASE TYPES

  // fwd StageFunctionTpl
  template<typename Scalar>
  struct StageFunctionTpl;

  // fwd FunctionDataTpl
  template<typename Scalar>
  struct FunctionDataTpl;

  // fwd CostBaseTpl
  template<typename Scalar>
  struct CostBaseTpl;

  template<typename Scalar>
  struct CostDataTpl;

  // fwd DynamicsModelTpl
  template<typename Scalar>
  struct DynamicsModelTpl;

  // fwd StageConstraintTpl
  template<typename Scalar>
  struct StageConstraintTpl;

  // fwd ExplicitDynamicsModelTpl
  template<typename Scalar>
  struct ExplicitDynamicsModelTpl;

  // fwd declaration of ExplicitDynamicsDataTpl
  template<typename _Scalar>
  struct ExplicitDynamicsDataTpl;


  /// STAGE MODEL

  // fwd StageModelTpl
  template<typename Scalar>
  class StageModelTpl;

  template<typename Scalar>
  struct StageDataTpl;


  /// SHOOTING PROBLEM

  // fwd ShootingProblemTpl
  template<typename _Scalar>
  struct ShootingProblemTpl;

  template<typename _Scalar>
  struct ProblemDataTpl;


  /// Math utilities
  namespace math
  {
    using namespace proxnlp::math;
  } // namespace math

} // namespace proxddp

