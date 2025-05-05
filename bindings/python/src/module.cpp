/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/utils.hpp"

#include "aligator/core/enums.hpp"
#include "aligator/threads.hpp"

#include <eigenpy/optional.hpp>

namespace aligator {
namespace python {

void exposeExplicitIntegrators();
#ifdef ALIGATOR_WITH_CROCODDYL_COMPAT
void exposeCrocoddylCompat();
#endif

static void exposeEnums() {
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
      .value("HESSIAN_BFGS", HessianApprox::BFGS)
      .export_values();

  bp::enum_<StepAcceptanceStrategy>("StepAcceptanceStrategy",
                                    "Step acceptance strategy.")
      .value("SA_LINESEARCH_ARMIJO", StepAcceptanceStrategy::LINESEARCH_ARMIJO)
      .value("SA_LINESEARCH_NONMONOTONE",
             StepAcceptanceStrategy::LINESEARCH_NONMONOTONE)
      .value("SA_FILTER", StepAcceptanceStrategy::FILTER)
      .export_values();
}

static void exposeContainers() {
  StdVectorPythonVisitor<std::vector<long>, true>::expose("StdVec_long");
  eigenpy::exposeStdVectorEigenSpecificType<context::Vector3s>(
      "StdVec_Vector3s");
  StdVectorPythonVisitor<std::vector<bool>, true>::expose("StdVec_bool");
}

/// Expose manifolds
void exposeManifolds();

} // namespace python
} // namespace aligator

BOOST_PYTHON_MODULE(MODULE_NAME) {
  using namespace aligator::python;
  using aligator::context::ConstVectorRef;

  bp::docstring_options module_docstring_options(true, true, true);

  bp::scope().attr("__version__") = ALIGATOR_VERSION;
#ifdef ALIGATOR_MULTITHREADING
  bp::def("get_available_threads", &aligator::omp::get_available_threads,
          "Get the number of available threads.");
  bp::def("get_current_threads", &aligator::omp::get_current_threads,
          "Get the current number of threads.");
  bp::def("set_omp_default_options", &aligator::omp::set_default_options,
          ("num_threads"_a, "dynamic"_a = true));
#endif
  eigenpy::enableEigenPy();

  eigenpy::OptionalConverter<ConstVectorRef, std::optional>::registration();
  eigenpy::detail::NoneToPython<std::nullopt_t>::registration();

  bp::import("warnings");
  bp::import("proxsuite_nlp");

  bp::def(
      "has_pinocchio_features",
      +[]() constexpr -> bool {
        return
#ifdef ALIGATOR_WITH_PINOCCHIO
            true;
#else
            false;
#endif
      },
      "Whether Aligator (and its Python bindings) were compiled with support "
      "for Pinocchio.");

  {
    bp::scope manifolds = get_namespace("manifolds");
    exposeManifolds();
  }
  exposeContainers();
  exposeGAR();
  exposeEnums();
  exposeContainers();
  exposeFunctions();
  exposeCosts();
  exposeConstraint();
  exposeStage();
  exposeProblem();
  exposeFilter();
  {
    bp::scope dynamics = get_namespace("dynamics");
    exposeContinuousDynamics();
    exposeDynamics();
    exposeExplicitIntegrators();
    exposeIntegrators();
  }
  exposeUtils();

  exposeSolvers();
  exposeCallbacks();
  exposeAutodiff();

#ifdef ALIGATOR_WITH_PINOCCHIO
  exposePinocchioFeatures();
#endif

#ifdef ALIGATOR_WITH_CROCODDYL_COMPAT
  {
    bp::scope croc_ns = get_namespace("croc");
    exposeCrocoddylCompat();
  }
#endif
}
