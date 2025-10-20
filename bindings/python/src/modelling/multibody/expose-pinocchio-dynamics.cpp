/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, 2023-2025 INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/utils.hpp"

namespace aligator {
namespace python {

#ifdef ALIGATOR_WITH_PINOCCHIO

//
// Factor out the Pinocchio dynamics
//

void exposeFreeFwdDynamics();
void exposeKinodynamics();
void exposeConstrainedFwdDynamics();

void exposePinocchioDynamics() {
  bp::scope dyn = get_namespace("dynamics");
  exposeFreeFwdDynamics();
  exposeKinodynamics();
  exposeConstrainedFwdDynamics();
}

#endif

} // namespace python
} // namespace aligator
