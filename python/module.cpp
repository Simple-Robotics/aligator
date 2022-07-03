#include "proxddp/python/fwd.hpp"
#include "proxddp/python/util.hpp"

#include "proxddp/utils.hpp"

namespace proxddp {
namespace python {
void exposeUtils() {
  using DynamicsType = DynamicsModelTpl<context::Scalar>;
  using ExplicitDynamics = ExplicitDynamicsModelTpl<context::Scalar>;

  using rollout_generic_t = context::VectorOfVectors (*)(
      const context::Manifold &, const DynamicsType &,
      const context::VectorXs &, const context::VectorOfVectors &);

  using rollout_explicit_t = context::VectorOfVectors (*)(
      const ExplicitDynamics &, const context::VectorXs &,
      const context::VectorOfVectors &);

  using rollout_vec_explicit_t = context::VectorOfVectors (*)(
      const std::vector<const ExplicitDynamics *> &, const context::VectorXs &,
      const context::VectorOfVectors &);

  bp::def<rollout_generic_t>(
      "rollout", &proxddp::rollout, bp::args("space", "dyn_model", "x0", "us"),
      "Perform a dynamics rollout, for a generic (implicit) dynamics model.");

  bp::def<rollout_explicit_t>(
      "rollout", &proxddp::rollout, bp::args("dyn_model", "x0", "us"),
      "Perform a rollout of a single explicit dynamics model.");

  bp::def<rollout_vec_explicit_t>(
      "rollout", &proxddp::rollout, bp::args("dyn_models", "x0", "us"),
      "Perform a rollout of an explicit dynamics model.");
}

} // namespace python
} // namespace proxddp

BOOST_PYTHON_MODULE(pyproxddp) {
  using namespace proxddp::python;

  bp::docstring_options module_docstring_options(true, true, true);

  eigenpy::enableEigenPy();

  bp::import("warnings");

  exposeFunctions();
  exposeCosts();
  exposeStage();
  exposeProblem();
  {
    bp::scope dynamics = get_namespace("dynamics");
    exposeODEs();
    exposeDynamics();
    exposeFreeFwdDynamics();
    exposeIntegrators();
  }
  exposeUtils();
  exposeSolvers();
}
