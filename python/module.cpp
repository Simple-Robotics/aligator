#include "proxddp/python/fwd.hpp"
#include "proxddp/python/util.hpp"

#include "proxddp/utils.hpp"


namespace proxddp
{
  namespace python
  {
    void exposeUtils()
    {
      using explicit_dyn_t = ExplicitDynamicsModelTpl<context::Scalar>;

      using rollout_explicit_t = context::VectorOfVectors(*)(
        const explicit_dyn_t&,
        const context::VectorXs&,
        const context::VectorOfVectors&
      );

      using rollout_vec_explicit_t = context::VectorOfVectors(*)(
        const std::vector<const explicit_dyn_t*>&,
        const context::VectorXs&,
        const context::VectorOfVectors&
      );

      bp::def("rollout",
              static_cast<rollout_explicit_t>(&proxddp::rollout),
              bp::args("dyn_model", "x0", "us"),
              "Perform a rollout of a single explicit dynamics model.");

      bp::def("rollout",
              static_cast<rollout_vec_explicit_t>(&proxddp::rollout),
              bp::args("dyn_models", "x0", "us"),
              "Perform a rollout of an explicit dynamics model.");

    }

  } // namespace python
} // namespace proxddp


BOOST_PYTHON_MODULE(pyproxddp)
{
  using namespace proxddp::python;

  bp::docstring_options module_docstring_options(true, true, true);

  eigenpy::enableEigenPy();

  bp::import("warnings");

  exposeFunctions();
  exposeNode();
  exposeProblem();
  {
    bp::scope dynamics = get_namespace("dynamics");
    exposeDynamics();
    exposeIntegrators();
  }
  exposeUtils();

}

