#pragma once

#include "aligator/context.hpp"
#include "aligator/python/macros.hpp"
#include "aligator/python/visitors.hpp"

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

namespace aligator {
/// @brief  The Python bindings.
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

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
void exposeODEs();
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
