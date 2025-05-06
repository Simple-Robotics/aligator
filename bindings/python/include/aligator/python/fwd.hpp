#pragma once

#include "aligator/fwd.hpp"
#include "aligator/context.hpp"

namespace boost {
template <typename T, typename A>
inline T *get_pointer(::xyz::polymorphic<T, A> const &x) {
  const T *r = x.operator->();
  return const_cast<T *>(r);
}
} // namespace boost

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

#include "aligator/python/polymorphic.hpp"

/// @brief  The Python bindings.
namespace aligator::python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

/// User-defined literal for defining boost::python::arg
inline bp::arg operator""_a(const char *argname, std::size_t) {
  return bp::arg(argname);
}

namespace internal {

template <typename ret_type>
ret_type suppress_if_void(bp::detail::method_result &&o) {
  if constexpr (!std::is_void_v<ret_type>)
    return o.operator ret_type();
}

} // namespace internal

/// Expose GAR module
void exposeGAR();
/// Expose stagewise function classes
void exposeFunctions();
/// Expose cost functions
void exposeCosts();
/// Expose constraints
void exposeConstraint();
/// Expose StageModel and StageData
void exposeStage();
/// Expose TrajOptProblem
void exposeProblem();

/// Expose discrete dynamics
void exposeDynamics();
/// Expose continuous dynamics
void exposeContinuousDynamics();
/// Expose numerical integrators
void exposeIntegrators();

/// Expose solvers
void exposeSolvers();
/// Expose solver callbacks
void exposeCallbacks();
/// Expose autodiff helpers
void exposeAutodiff();
/// Expose utils
void exposeUtils();
/// Expose filter strategy
void exposeFilter();

#ifdef ALIGATOR_WITH_PINOCCHIO
/// Expose features using the Pinocchio rigid dynamics library
void exposePinocchioFeatures();
#endif

} // namespace aligator::python

#define ALIGATOR_PYTHON_OVERRIDE_IMPL(ret_type, pyname, ...)                   \
  do {                                                                         \
    if (bp::override fo = this->get_override(pyname)) {                        \
      decltype(auto) o = fo(__VA_ARGS__);                                      \
      return ::aligator::python::internal::suppress_if_void<ret_type>(         \
          std::move(o));                                                       \
    }                                                                          \
  } while (false)

/**
 * @def ALIGATOR_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)
 * @brief Define the body of a virtual function override. This is meant
 *        to reduce boilerplate code when exposing virtual member functions.
 */
#define ALIGATOR_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)                   \
  ALIGATOR_PYTHON_OVERRIDE_IMPL(ret_type, pyname, __VA_ARGS__);                \
  ALIGATOR_RUNTIME_ERROR("Tried to call pure virtual function {:s}.", pyname)

/**
 * @def ALIGATOR_PYTHON_OVERRIDE(ret_type, cname, fname, ...)
 * @copybrief ALIGATOR_PYTHON_OVERRIDE_PURE()
 */
#define ALIGATOR_PYTHON_OVERRIDE(ret_type, cname, fname, ...)                  \
  ALIGATOR_PYTHON_OVERRIDE_IMPL(ret_type, #fname, __VA_ARGS__);                \
  return cname::fname(__VA_ARGS__)
