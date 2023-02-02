/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/utils.hpp"

#include "proxddp/version.hpp"
#include "proxddp/threads.hpp"

#ifdef PROXDDP_WITH_CROCODDYL_COMPAT
#include "proxddp/python/compat/croco.hpp"
#endif

BOOST_PYTHON_MODULE(pyproxddp) {
  using namespace proxddp::python;

  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = proxddp::printVersion();
  bp::def("get_available_threads", &proxddp::omp::get_available_threads,
          "Get the number of available threads.");
  bp::def("get_current_threads", &proxddp::omp::get_current_threads,
          "Get the current number of threads.");
  eigenpy::enableEigenPy();

  bp::import("warnings");
  bp::import("proxnlp");

  exposeFunctions();
#ifdef PROXDDP_WITH_PINOCCHIO
  exposePinocchioFunctions();
#endif
  exposeCosts();
  exposeStage();
  exposeProblem();
  {
    bp::scope dynamics = get_namespace("dynamics");
    exposeODEs();
    exposeDynamics();

#ifdef PROXDDP_WITH_PINOCCHIO
    exposeFreeFwdDynamics();
#ifdef PROXDDP_PINOCCHIO_V3
    exposeConstraintFwdDynamics();
#endif
#endif

    exposeIntegrators();
  }
  exposeUtils();

  exposeSolvers();
  exposeCallbacks();
  exposeAutodiff();

#ifdef PROXDDP_WITH_CROCODDYL_COMPAT
  {
    bp::scope croc_ns = get_namespace("croc");
    exposeCrocoddylCompat();
  }
#endif
}
