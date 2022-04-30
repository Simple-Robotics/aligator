#include "proxddp/python/fwd.hpp"
#include "proxddp/python/util.hpp"



namespace proxddp
{
  namespace python
  {

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
    exposeIntegrators();
  }

}

