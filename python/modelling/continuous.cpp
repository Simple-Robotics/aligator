#include "proxddp/python/fwd.hpp"
#include "proxddp/python/modelling/continuous.hpp"


namespace proxddp
{
  namespace python
  {
    void exposeODEs()
    {
      using namespace proxddp::dynamics;
      using context::Scalar;
      using ManifoldPtr = shared_ptr<context::Manifold>;
      using ContinuousDynamicsBase = ContinuousDynamicsAbstractTpl<Scalar>;
      using ContinuousDynamicsData = ContinuousDynamicsDataTpl<Scalar>;
      using ODEAbstract = ODEAbstractTpl<Scalar>;
      using ODEData = ODEDataTpl<Scalar>;
      using internal::PyContinuousDynamics;
      using internal::PyODEAbstract;

      bp::register_ptr_to_python< shared_ptr<PyContinuousDynamics<>> >();
      bp::class_<PyContinuousDynamics<>, boost::noncopyable>(
        "ContinuousDynamicsBase",
        "Base class for continuous-time dynamical models (DAEs and ODEs).",
        bp::init<const ManifoldPtr&, int>(
          "Default constructor: provide the working manifold and control space dimension.",
          bp::args("self", "space", "nu"))
      )
        .add_property("ndx", &ContinuousDynamicsBase::ndx, "State space dimension.")
        .add_property("nu",  &ContinuousDynamicsBase::nu, "Control space dimension.")
        .def("evaluate",
             bp::pure_virtual(&ContinuousDynamicsBase::evaluate),
             bp::args("self", "x", "u", "xdot", "data"),
             "Evaluate the DAE functions.")
        .def("computeJacobians",  
             bp::pure_virtual(&ContinuousDynamicsBase::computeJacobians),
             bp::args("self", "x", "u", "xdot", "data"),
             "Evaluate the DAE function derivatives.") 
        .def(CreateDataPythonVisitor<ContinuousDynamicsBase>())
        ;

      bp::register_ptr_to_python<shared_ptr<ContinuousDynamicsData>>();
      bp::class_<ContinuousDynamicsData>(
        "ContinuousDynamicsData",
        "Data struct for continuous dynamics/DAE models.",
        bp::init<int, int>(bp::args("self", "ndx", "nu")))
        .add_property("value",
                      bp::make_getter(&ContinuousDynamicsData::value_, bp::return_internal_reference<>()),
                      "Vector value of the DAE residual.")
        .add_property("Jx",
                      bp::make_getter(&ContinuousDynamicsData::Jx_, bp::return_internal_reference<>()),
                      "Jacobian with respect to state.")
        .add_property("Ju",
                      bp::make_getter(&ContinuousDynamicsData::Ju_, bp::return_internal_reference<>()),
                      "Jacobian with respect to controls.")
        .add_property("Jxdot",
                      bp::make_getter(&ContinuousDynamicsData::Jxdot_, bp::return_internal_reference<>()),
                      "Jacobian with respect to :math:`\\dot{x}`.")
        ;

      bp::register_ptr_to_python< shared_ptr<ODEAbstract> >();
      bp::class_<PyODEAbstract, bp::bases<ContinuousDynamicsBase>, boost::noncopyable>(
        "ODEAbstract", "Continuous dynamics described by ordinary differential equations (ODEs).",
        bp::init<const ManifoldPtr&, int>(bp::args("self", "space", "nu"))
      )
        .def("forward",  bp::pure_virtual(&ODEAbstract::forward),
             bp::args("self", "x", "u", "data"),
             "Compute the value of the ODE vector field, i.e. the "
             "state time derivative :math:`\\dot{x}`.")
        .def("dForward", bp::pure_virtual(&ODEAbstract::dForward),
             bp::args("self", "x", "u", "data"),
             "Compute the derivatives of the ODE vector field with respect "
             "to the state-control pair :math:`(x, u)`.")
        .def(CreateDataPythonVisitor<ODEAbstract>())
        ;
      
      bp::register_ptr_to_python<shared_ptr<ODEData>>();
      bp::class_<ODEData, bp::bases<ContinuousDynamicsData>>(
        "ODEData", "Data struct for ODE models.", bp::init<int, int>(bp::args("self", "ndx", "nu")))
        .add_property("xdot", bp::make_getter(&ODEData::xdot_, bp::return_internal_reference<>()));

    }

  } // namespace python 
} // namespace proxddp
