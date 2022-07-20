#pragma once

#include "proxddp/python/context.hpp"
#include "proxddp/python/macros.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

#include <eigenpy/eigenpy.hpp>
#include "proxddp/python/visitors.hpp"

namespace proxddp {
/// @brief  The Python bindings.
namespace python {
namespace pinpy = pinocchio::python;
namespace bp = boost::python;

/// Expose ternary functions
void exposeFunctions();
void exposeCosts();
void exposeStage();
void exposeProblem();

/// Expose continuous dynamics models.
void exposeODEs();
void exposeFreeFwdDynamics();
void exposeConstraintFwdDynamics();
void exposeDynamics();
/// Expose integrators
void exposeIntegrators();
void exposeSolvers();
void exposeCallbacks();

} // namespace python
} // namespace proxddp
