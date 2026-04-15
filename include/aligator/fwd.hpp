/// @file fwd.hpp
/// @brief Forward declarations.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#ifdef EIGEN_DEFAULT_IO_FORMAT
#undef EIGEN_DEFAULT_IO_FORMAT
#endif
#define EIGEN_DEFAULT_IO_FORMAT                                                \
  Eigen::IOFormat(Eigen::StreamPrecision, 0, ",", "\n", "[", "]")

#include "aligator/math.hpp"
#include "aligator/utils/exceptions.hpp"
#include "aligator/config.hpp"
#include "aligator/deprecated.hpp"

#include <memory>

#define ALIGATOR_RAISE_IF_NAN(value)                                           \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN.\n")

#define ALIGATOR_RAISE_IF_NAN_NAME(value, name)                                \
  if (::aligator::math::check_value(value))                                    \
  ALIGATOR_RUNTIME_ERROR("Encountered NaN for variable {:s}\n", name)

#define ALIGATOR_INLINE inline __attribute__((always_inline))

/// \brief macros for pragma push/pop/ignore deprecated warnings
#if defined(__GNUC__) || defined(__clang__)
#define ALIGATOR_COMPILER_DIAGNOSTIC_PUSH ALIGATOR_PRAGMA(GCC diagnostic push)
#define ALIGATOR_COMPILER_DIAGNOSTIC_POP ALIGATOR_PRAGMA(GCC diagnostic pop)
#if defined(__clang__)
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
ALIGATOR_PRAGMA(GCC diagnostic ignored "-Wdelete-non-abstract-non-virtual-dtor")
#else
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
#endif
#elif defined(WIN32)
#define ALIGATOR_COMPILER_DIAGNOSTIC_PUSH _Pragma("warning(push)")
#define ALIGATOR_COMPILER_DIAGNOSTIC_POP _Pragma("warning(pop)")
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
#else
#define ALIGATOR_COMPILER_DIAGNOSTIC_PUSH
#define ALIGATOR_COMPILER_DIAGNOSTIC_POP
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_DEPRECECATED_DECLARATIONS
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_VARIADIC_MACROS
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_SELF_ASSIGN_OVERLOADED
#define ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_MAYBE_UNINITIALIZED
#endif // __GNUC__ || __clang__

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

} // namespace aligator
