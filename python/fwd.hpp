#pragma once

#include "proxddp/context.hpp"
#include "proxddp/python/macros.hpp"
#include "proxddp/python/visitors.hpp"

#ifdef PROXDDP_WITH_PINOCCHIO
#include <pinocchio/fwd.hpp>
#endif

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

#if PINOCCHIO_VERSION_AT_LEAST(2, 9, 2)
#define PROXDDP_PINOCCHIO_V3
#endif

namespace proxddp {
/// @brief  The Python bindings.
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

/// Expose ternary functions
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
/// Expose integrators
void exposeIntegrators();

/// Expose solvers
void exposeSolvers();
void exposeCallbacks();
void exposeAutodiff();
void exposeUtils();

#ifdef PROXDDP_WITH_PINOCCHIO
void exposePinocchioFeatures();
#endif

} // namespace python
} // namespace proxddp
