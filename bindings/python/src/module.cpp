/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/utils.hpp"

#include "proxddp/core/enums.hpp"
#include "proxddp/version.hpp"
#include "proxddp/threads.hpp"

#ifdef PROXDDP_WITH_CROCODDYL_COMPAT
#include "proxddp/python/compat/croco.hpp"
#endif

namespace aligator {
namespace python {
void exposeEnums() {
  register_enum_symlink<VerboseLevel>(true);

  bp::enum_<MultiplierUpdateMode>(
      "MultiplierUpdateMode", "Enum for the kind of multiplier update to use.")
      .value("NEWTON", MultiplierUpdateMode::NEWTON)
      .value("PRIMAL", MultiplierUpdateMode::PRIMAL)
      .value("PRIMAL_DUAL", MultiplierUpdateMode::PRIMAL_DUAL);

  bp::enum_<LinesearchMode>("LinesearchMode", "Linesearch mode.")
      .value("PRIMAL", LinesearchMode::PRIMAL)
      .value("PRIMAL_DUAL", LinesearchMode::PRIMAL_DUAL);

  bp::enum_<RolloutType>("RolloutType", "Rollout type.")
      .value("ROLLOUT_LINEAR", RolloutType::LINEAR)
      .value("ROLLOUT_NONLINEAR", RolloutType::NONLINEAR)
      .export_values();

  bp::enum_<HessianApprox>("HessianApprox",
                           "Level of approximation for the Hessian.")
      .value("HESSIAN_EXACT", HessianApprox::EXACT)
      .value("HESSIAN_GAUSS_NEWTON", HessianApprox::GAUSS_NEWTON)
      .export_values();
}

} // namespace python
} // namespace aligator

BOOST_PYTHON_MODULE(MODULE_NAME) {
  using namespace aligator::python;

  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = aligator::printVersion();
#ifdef PROXDDP_MULTITHREADING
  bp::def("get_available_threads", &aligator::omp::get_available_threads,
          "Get the number of available threads.");
  bp::def("get_current_threads", &aligator::omp::get_current_threads,
          "Get the current number of threads.");
#endif
  eigenpy::enableEigenPy();

  bp::import("warnings");
  bp::import("proxsuite_nlp");

  exposeEnums();
  exposeFunctions();
  exposeCosts();
  exposeConstraint();
  exposeStage();
  exposeProblem();
  {
    bp::scope dynamics = get_namespace("dynamics");
    exposeODEs();
    exposeDynamics();
    exposeIntegrators();
  }
  exposeUtils();

  exposeSolvers();
  exposeCallbacks();
  exposeAutodiff();

#ifdef PROXDDP_WITH_PINOCCHIO
  exposePinocchioFeatures();
#endif

#ifdef PROXDDP_WITH_CROCODDYL_COMPAT
  {
    bp::scope croc_ns = get_namespace("croc");
    exposeCrocoddylCompat();
  }
#endif
}
