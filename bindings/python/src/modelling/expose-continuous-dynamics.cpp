/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/python/modelling/continuous.hpp"
#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"
#include "aligator/modelling/dynamics/context.hpp"

namespace aligator {
namespace python {
using namespace ::aligator::dynamics;
using context::MatrixXs;
using context::Scalar;
using context::VectorXs;

using context::ContinuousDynamicsAbstract;
using context::ContinuousDynamicsData;

struct ContinousDataWrapper : ContinuousDynamicsData,
                              bp::wrapper<ContinuousDynamicsData> {
  using ContinuousDynamicsData::ContinuousDynamicsData;
};

void exposeODEs();

void exposeContinuousDynamics() {
  using ManifoldPtr = xyz::polymorphic<context::Manifold>;

  register_polymorphic_to_python<
      xyz::polymorphic<ContinuousDynamicsAbstract>>();
  PolymorphicMultiBaseVisitor<ContinuousDynamicsAbstract> conversion_visitor;

  bp::class_<PyContinuousDynamics<>, boost::noncopyable>(
      "ContinuousDynamicsAbstract",
      "Base class for continuous-time dynamical models (DAEs and ODEs).",
      bp::init<ManifoldPtr, int>("Default constructor: provide the working "
                                 "manifold and control space "
                                 "dimension.",
                                 bp::args("self", "space", "nu")))
      .add_property("ndx", &ContinuousDynamicsAbstract::ndx,
                    "State space dimension.")
      .add_property("nu", &ContinuousDynamicsAbstract::nu,
                    "Control space dimension.")
      .def("evaluate", bp::pure_virtual(&ContinuousDynamicsAbstract::evaluate),
           bp::args("self", "x", "u", "xdot", "data"),
           "Evaluate the DAE functions.")
      .def("computeJacobians",
           bp::pure_virtual(&ContinuousDynamicsAbstract::computeJacobians),
           bp::args("self", "x", "u", "xdot", "data"),
           "Evaluate the DAE function derivatives.")
      .add_property("space",
                    bp::make_function(&ContinuousDynamicsAbstract::space,
                                      bp::return_internal_reference<>()),
                    "Get the state space.")
      .def(CreateDataPolymorphicPythonVisitor<ContinuousDynamicsAbstract,
                                              PyContinuousDynamics<>>())
      .def(conversion_visitor);

  bp::register_ptr_to_python<shared_ptr<ContinuousDynamicsData>>();
  auto cont_data_cls =
      bp::class_<ContinousDataWrapper, boost::noncopyable>(
          "ContinuousDynamicsData",
          "Data struct for continuous dynamics/DAE models.",
          bp::init<int, int>(bp::args("self", "ndx", "nu")))
          .add_property("value",
                        bp::make_getter(&ContinuousDynamicsData::value_,
                                        bp::return_internal_reference<>()),
                        "Vector value of the DAE residual.")
          .add_property("Jx",
                        bp::make_getter(&ContinuousDynamicsData::Jx_,
                                        bp::return_internal_reference<>()),
                        "Jacobian with respect to state.")
          .add_property("Ju",
                        bp::make_getter(&ContinuousDynamicsData::Ju_,
                                        bp::return_internal_reference<>()),
                        "Jacobian with respect to controls.")
          .add_property("Jxdot",
                        bp::make_getter(&ContinuousDynamicsData::Jxdot_,
                                        bp::return_internal_reference<>()),
                        "Jacobian with respect to :math:`\\dot{x}`.")
          .add_property("xdot",
                        bp::make_getter(&ContinuousDynamicsData::xdot_,
                                        bp::return_internal_reference<>()),
                        "Time derivative :math:`\\dot{x}`.");

  // Alias this for back-compatibility
  bp::scope().attr("ODEData") = cont_data_cls;

  exposeODEs();
}

} // namespace python
} // namespace aligator
