#pragma once

#include "proxddp/python/context.hpp"
#include "proxddp/python/macros.hpp"
#include "proxddp/python/visitors.hpp"

#include <pinocchio/fwd.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

namespace proxddp {
/// @brief  The Python bindings.
namespace python {
namespace bp = boost::python;
using eigenpy::StdVectorPythonVisitor;

/// Expose ternary functions
void exposeFunctions();
void exposeCosts();
void exposeStage();
void exposeProblem();

/// Expose continuous dynamics models.
void exposeODEs();
void exposeDynamics();
/// Expose integrators
void exposeIntegrators();

/// Expose solvers
void exposeSolvers();
void exposeCallbacks();
void exposeAutodiff();
void exposeUtils();

#ifdef PROXDDP_WITH_PINOCCHIO
void exposePinocchioFunctions();
void exposeFreeFwdDynamics();
#if PINOCCHIO_VERSION_AT_LEAST(2, 9, 2)
void exposeConstraintFwdDynamics();
#endif
#endif

} // namespace python
} // namespace proxddp
