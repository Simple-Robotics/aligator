#pragma once
/// @file fwd.hpp
/// @brief Forward declarations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <proxnlp/fwd.hpp>

#ifdef PROXDDP_WITH_PINOCCHIO
#include "pinocchio/container/boost-container-limits.hpp"
#endif

/// @brief  Main package namespace.
namespace proxddp {
/// TYPEDEFS FROM PROXNLP

// Use the shared_ptr used in proxnlp.
using proxnlp::ManifoldAbstractTpl;
// Use the math_types template from proxnlp.
using proxnlp::math_types;
// Using the constraint set from proxnlp
using proxnlp::ConstraintSetBase;
using proxnlp::VerboseLevel;

using VerboseLevel::QUIET;
using VerboseLevel::VERBOSE;
using VerboseLevel::VERYVERBOSE;

using std::shared_ptr;
using std::unique_ptr;

/// 1 BASE TYPES

// fwd StageFunctionTpl
template <typename Scalar> struct StageFunctionTpl;

// fwd FunctionDataTpl
template <typename Scalar> struct FunctionDataTpl;

// fwd CostAbstractTpl
template <typename Scalar> struct CostAbstractTpl;

// fwd CostDataAbstractTpl
template <typename Scalar> struct CostDataAbstractTpl;

// fwd DynamicsModelTpl
template <typename Scalar> struct DynamicsModelTpl;

// fwd DynamicsDataTpl
template <typename Scalar> using DynamicsDataTpl = FunctionDataTpl<Scalar>;

// fwd StageConstraintTpl
template <typename Scalar> struct StageConstraintTpl;

// fwd ExplicitDynamicsModelTpl
template <typename Scalar> struct ExplicitDynamicsModelTpl;

// fwd declaration of ExplicitDynamicsDataTpl
template <typename Scalar> struct ExplicitDynamicsDataTpl;

/* FUNCTION EXPRESSIONS */

// fwd declaration of FunctionSliceXprTpl
template <typename Scalar> struct FunctionSliceXprTpl;

/* STAGE MODEL */

// fwd StageModelTpl
template <typename Scalar> struct StageModelTpl;

template <typename Scalar> struct StageDataTpl;

/* SHOOTING PROBLEM */

// fwd TrajOptProblemTpl
template <typename Scalar> struct TrajOptProblemTpl;

// fwd TrajOptDataTpl
template <typename Scalar> struct TrajOptDataTpl;

// fwd SolverProxDDP
template <typename Scalar> struct SolverProxDDP;

// fwd SolverFDDP
template <typename Scalar> struct SolverFDDP;

// fwd WorkspaceBaseTpl
template <typename Scalar> struct WorkspaceBaseTpl;

// fwd ResultsBaseTpl
template <typename Scalar> struct ResultsBaseTpl;

// fwd WorkspaceTpl
template <typename Scalar> struct WorkspaceTpl;

// fwd ResultsTpl
template <typename Scalar> struct ResultsTpl;

} // namespace proxddp

#include "proxddp/math.hpp"
#include "proxddp/macros.hpp"
#include "proxddp/config.hpp"
