#pragma once

#include "aligator/context.hpp"
#include "aligator/python/macros.hpp"

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

#include <proxsuite-nlp/python/polymorphic.hpp>

namespace aligator {
/// @brief  The Python bindings.
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

/// User-defined literal for defining boost::python::arg
inline bp::arg operator""_a(const char *argname, std::size_t) {
  return bp::arg(argname);
}

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
void exposeUtils();
void exposeFilter();

#ifdef ALIGATOR_WITH_PINOCCHIO
/// Expose features using the Pinocchio rigid dynamics library
void exposePinocchioFeatures();
#endif

} // namespace python
} // namespace aligator
