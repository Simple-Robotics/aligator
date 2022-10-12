/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/utils.hpp"

#include "proxddp/version.hpp"

#ifdef WITH_CROCODDYL_COMPAT
#include "proxddp/python/compat/croco.hpp"
#endif

BOOST_PYTHON_MODULE(pyproxddp) {
  using namespace proxddp::python;

  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = proxddp::printVersion();
  eigenpy::enableEigenPy();

  bp::import("warnings");
  bp::import("proxnlp");

  exposeFunctions();
  exposeCosts();
  exposeStage();
  exposeProblem();
  {
    bp::scope dynamics = get_namespace("dynamics");
    exposeODEs();
    exposeDynamics();

#ifdef PROXDDP_WITH_PINOCCHIO
    exposeFreeFwdDynamics();
    exposeConstraintFwdDynamics();
#endif

    exposeIntegrators();
  }
  exposeUtils();
  exposeSolvers();
  exposeCallbacks();
  exposeAutodiff();

#ifdef WITH_CROCODDYL_COMPAT
  exposeCrocoddylCompat();
#endif
}
