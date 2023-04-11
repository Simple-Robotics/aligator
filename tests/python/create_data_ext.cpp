#include "proxddp/core/explicit-dynamics.hpp"
#include <eigenpy/eigenpy.hpp>

using proxddp::context::ExplicitDynamics;

auto my_create_data(const ExplicitDynamics &dyn) { return dyn.createData(); }

BOOST_PYTHON_MODULE(create_data_ext) {
  namespace bp = boost::python;
  bp::def("my_create_data", my_create_data, bp::args("dyn"));
}
