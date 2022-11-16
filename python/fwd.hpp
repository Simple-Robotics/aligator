#pragma once

#include "proxddp/python/context.hpp"
#include "proxddp/python/macros.hpp"
#include "proxddp/python/visitors.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

#include <eigenpy/eigenpy.hpp>

namespace proxddp {
/// @brief  The Python bindings.
namespace python {
namespace pp = pinocchio::python;
namespace bp = boost::python;

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
void exposeSolvers();
void exposeFDDP();
void exposeCallbacks();
void exposeAutodiff();
void exposeUtils();

#ifdef PROXDDP_WITH_PINOCCHIO
void exposePinocchioFunctions();
void exposeFreeFwdDynamics();
void exposeConstraintFwdDynamics();
#endif

} // namespace python
} // namespace proxddp
