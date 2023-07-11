/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/utils.hpp"

namespace proxddp {
namespace python {

#ifdef PROXDDP_WITH_PINOCCHIO

void exposePinocchioFunctions();
void exposeFreeFwdDynamics();
#ifdef PROXDDP_PINOCCHIO_V3
void exposeConstrainedFwdDynamics();
#endif

void exposePinocchioFeatures() {
  bp::import("pinocchio");
  exposePinocchioFunctions();

  {
    bp::scope dyn = get_namespace("dynamics");
    exposeFreeFwdDynamics();

#ifdef PROXDDP_PINOCCHIO_V3
    exposeConstrainedFwdDynamics();
#endif
  }
}

#endif

} // namespace python
} // namespace proxddp
