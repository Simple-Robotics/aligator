/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/utils.hpp"

namespace aligator {
namespace python {

#ifdef ALIGATOR_WITH_PINOCCHIO

void exposePinocchioFunctions();
void exposeFreeFwdDynamics();
void exposeKinodynamics();
void exposeConstrainedFwdDynamics();

void exposePinocchioFeatures() {
  bp::import("pinocchio");
  exposePinocchioFunctions();

  {
    bp::scope dyn = get_namespace("dynamics");
    exposeFreeFwdDynamics();
    exposeKinodynamics();
    exposeConstrainedFwdDynamics();
  }
}

#endif

} // namespace python
} // namespace aligator
