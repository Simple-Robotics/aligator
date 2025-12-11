/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/utils.hpp"
#include "aligator/python/string-view-converter.hpp"

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
#define _c(Enum, name) value(#name, Enum::name)

  bp::enum_<VerboseLevel>("VerboseLevel",
                          "Verbosity level to be used in solvers.")
      ._c(VerboseLevel, QUIET)
      ._c(VerboseLevel, VERBOSE)
      ._c(VerboseLevel, VERYVERBOSE)
      .export_values();

  bp::enum_<MultiplierUpdateMode>(
      "MultiplierUpdateMode", "Enum for the kind of multiplier update to use.")
      ._c(MultiplierUpdateMode, NEWTON)
      ._c(MultiplierUpdateMode, PRIMAL)
      ._c(MultiplierUpdateMode, PRIMAL_DUAL);

  bp::enum_<LinesearchMode>("LinesearchMode", "Linesearch mode.")
      ._c(LinesearchMode, PRIMAL)
      ._c(LinesearchMode, PRIMAL_DUAL);

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

#undef _c
}

static void exposeContainers() {
  using VecXBool = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
  using context::MatrixRef;
  using context::Scalar;
  using context::VectorRef;

  eigenpy::StdContainerFromPythonList<
      std::vector<std::string>>::register_converter();
  StdVectorPythonVisitor<std::vector<long>, true>::expose("StdVec_long");
  eigenpy::exposeStdVectorEigenSpecificType<context::Vector3s>(
      "StdVec_Vector3s");
  StdVectorPythonVisitor<std::vector<bool>, true>::expose("StdVec_bool");
  StdVectorPythonVisitor<std::vector<int>, true>::expose("StdVec_int");
  StdVectorPythonVisitor<std::vector<Scalar>, true>::expose("StdVec_Scalar");
  StdVectorPythonVisitor<context::VectorOfVectors, true>::expose(
      "StdVec_Vector");
  StdVectorPythonVisitor<std::vector<context::MatrixXs>, true>::expose(
      "StdVec_Matrix");
  StdVectorPythonVisitor<std::vector<VecXBool>, false>::expose(
      "StdVec_VecBool");
  StdVectorPythonVisitor<std::vector<VectorRef>, true>::expose("StdVec_VecRef");
  StdVectorPythonVisitor<std::vector<MatrixRef>, true>::expose("StdVec_MatRef");
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

  register_string_view_converter();
  eigenpy::StdVectorPythonVisitor<std::vector<std::string_view>, true>::expose(
      "StdVec_StringView");

  bp::import("warnings");

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
  {
    bp::import("pinocchio");
    exposePinocchioSpaces();
    exposePinocchioFunctions();
    exposePinocchioDynamics();
  }
#endif

#ifdef ALIGATOR_WITH_CROCODDYL_COMPAT
  {
    bp::scope croc_ns = get_namespace("croc");
    exposeCrocoddylCompat();
  }
#endif
}
