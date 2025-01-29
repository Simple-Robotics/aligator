/// @file fwd.hpp
/// @brief Forward declarations.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <proxsuite-nlp/fwd.hpp>
#include <proxsuite-nlp/config.hpp>

#ifdef ALIGATOR_WITH_PINOCCHIO
#include <pinocchio/config.hpp>
#endif

#include "aligator/math.hpp"
#include "aligator/utils/exceptions.hpp"
#include "aligator/macros.hpp"
#include "aligator/eigen-macros.hpp"
#include "aligator/config.hpp"
#include "aligator/deprecated.hpp"

#define ALIGATOR_RAISE_IF_NAN(value)                                           \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN.\n")

#define ALIGATOR_RAISE_IF_NAN_NAME(value, name)                                \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN for variable {:s}\n", name)

/// @brief  Main package namespace.
namespace aligator {
/// TYPEDEFS FROM PROXNLP

// NOLINTBEGIN(misc-unused-using-decls)

using proxsuite::nlp::ConstraintSetTpl;
using proxsuite::nlp::ManifoldAbstractTpl;
// Use the math_types template from proxsuite-nlp.
using proxsuite::nlp::VerboseLevel;

using VerboseLevel::QUIET;
using VerboseLevel::VERBOSE;
using VerboseLevel::VERYVERBOSE;

using std::shared_ptr;

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
template <typename Scalar> struct DynamicsDataTpl;

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
template <typename Scalar> struct SolverProxDDPTpl;

// fwd SolverFDDP
template <typename Scalar> struct SolverFDDPTpl;

// fwd WorkspaceBaseTpl
template <typename Scalar> struct WorkspaceBaseTpl;

// fwd ResultsBaseTpl
template <typename Scalar> struct ResultsBaseTpl;

// fwd WorkspaceTpl
template <typename Scalar> struct WorkspaceTpl;

// fwd ResultsTpl
template <typename Scalar> struct ResultsTpl;

// fwd FilterTpl
template <typename Scalar> struct FilterTpl;

template <typename T>
using StdVectorEigenAligned ALIGATOR_DEPRECATED_MESSAGE(
    "Aligator now requires C++17 and the Eigen::aligned_allocator<T> class is "
    "no longer useful. Please use std::vector<T> instead, this typedef will "
    "change to be an alias of that of the future, then will be removed.") =
    std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T, typename... Args>
ALIGATOR_DEPRECATED_MESSAGE(
    "Aligator now requires C++17 and the Eigen::aligned_allocator<T> class is "
    "no longer useful. This function is now just an alias for "
    "std::make_shared, and will be removed in the future.")
inline auto allocate_shared_eigen_aligned(Args &&...args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

} // namespace aligator
