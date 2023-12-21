#pragma once
/// @file fwd.hpp
/// @brief Forward declarations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <proxsuite-nlp/fwd.hpp>

#ifdef PROXDDP_WITH_PINOCCHIO
#include <pinocchio/fwd.hpp>
#if PINOCCHIO_VERSION_AT_LEAST(2, 9, 2)
#define PROXDDP_PINOCCHIO_V3
#endif
#endif

/// @brief  Main package namespace.
namespace aligator {
/// TYPEDEFS FROM PROXNLP

// NOLINTBEGIN(misc-unused-using-decls)

// Use the shared_ptr used in proxsuite-nlp.
using proxsuite::nlp::BCLParamsTpl;
using proxsuite::nlp::ConstraintSetBase;
using proxsuite::nlp::ManifoldAbstractTpl;
// Use the math_types template from proxsuite-nlp.
using proxsuite::nlp::VerboseLevel;

using VerboseLevel::QUIET;
using VerboseLevel::VERBOSE;
using VerboseLevel::VERYVERBOSE;

using std::shared_ptr;
using std::unique_ptr;

// NOLINTEND(misc-unused-using-decls)

// fwd StageFunctionTpl
template <typename Scalar> struct StageFunctionTpl;

// fwd UnaryFunctionTpl
template <typename Scalar> struct UnaryFunctionTpl;

// fwd StageFunctionDataTpl
template <typename Scalar> struct StageFunctionDataTpl;

// fwd CostAbstractTpl
template <typename Scalar> struct CostAbstractTpl;

// fwd CostDataAbstractTpl
template <typename Scalar> struct CostDataAbstractTpl;

// fwd DynamicsModelTpl
template <typename Scalar> struct DynamicsModelTpl;

// fwd DynamicsDataTpl
template <typename Scalar> using DynamicsDataTpl = StageFunctionDataTpl<Scalar>;

// fwd StageConstraintTpl
template <typename Scalar> struct StageConstraintTpl;

// fwd ExplicitDynamicsModelTpl
template <typename Scalar> struct ExplicitDynamicsModelTpl;

// fwd declaration of ExplicitDynamicsDataTpl
template <typename Scalar> struct ExplicitDynamicsDataTpl;

/* FUNCTION EXPRESSIONS */

// fwd declaration of FunctionSliceXprTpl
template <typename Scalar, typename Base> struct FunctionSliceXprTpl;

/* STAGE MODEL */

// fwd StageModelTpl
template <typename Scalar> struct StageModelTpl;

// fwd StageDataTpl
template <typename Scalar> struct StageDataTpl;

// fwd CallbackBaseTpl
template <typename Scalar> struct CallbackBaseTpl;

/* SHOOTING PROBLEM */

// fwd ConstraintStackTpl
template <typename Scalar> struct ConstraintStackTpl;

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

template <typename T>
using StdVectorEigenAligned = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T, typename... Args>
inline auto allocate_shared_eigen_aligned(Args &&...args) {
  return std::allocate_shared<T>(Eigen::aligned_allocator<T>(),
                                 std::forward<Args>(args)...);
}

} // namespace aligator

#include "proxddp/math.hpp"
#include "proxddp/macros.hpp"
#include "proxddp/config.hpp"
#include "proxddp/deprecated.hpp"
