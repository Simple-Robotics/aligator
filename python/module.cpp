#include "proxddp/python/fwd.hpp"



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

  exposeNode();

}

