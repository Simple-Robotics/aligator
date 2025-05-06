/// @file fwd.hpp
/// @brief Forward declarations.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/math.hpp"
#include "aligator/utils/exceptions.hpp"
#include "aligator/config.hpp"
#include "aligator/deprecated.hpp"

#define ALIGATOR_RAISE_IF_NAN(value)                                           \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN.\n")

#define ALIGATOR_RAISE_IF_NAN_NAME(value, name)                                \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN for variable {:s}\n", name)

#define ALIGATOR_INLINE inline __attribute__((always_inline))

namespace xyz {
// fwd-decl for boost override later
template <class T, class A> class polymorphic;
} // namespace xyz

/// The following overload for get_pointer is defined here, to avoid conflicts
/// with other Boost libraries using get_pointer() without seeing this overload
/// if included later.

/// @brief  Main package namespace.
namespace aligator {

template <typename Base, typename U, typename A = std::allocator<U>>
using is_polymorphic_of = std::is_same<std::decay_t<U>, xyz::polymorphic<U, A>>;

template <typename Base, typename U, typename A = std::allocator<U>>
constexpr bool is_polymorphic_of_v = is_polymorphic_of<Base, U, A>::value;

// NOLINTBEGIN(misc-unused-using-decls)

// fwd ManifoldAbstractTpl
template <typename Scalar> struct ManifoldAbstractTpl;

// fwd VectorSpaceTpl
template <typename Scalar, int Dim = Eigen::Dynamic> struct VectorSpaceTpl;

// fwd ConstraintSetTpl
template <typename Scalar> struct ConstraintSetTpl;

/// @brief Verbosity level.
enum VerboseLevel { QUIET = 0, VERBOSE = 1, VERYVERBOSE = 2 };

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

} // namespace aligator
