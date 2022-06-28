#include "proxddp/python/fwd.hpp"
#include "proxddp/python/dynamics-continuous.hpp"


namespace proxddp
{
  namespace python
  {
    void exposeODEs()
    {
      using namespace proxddp::dynamics;
      using context::Scalar;
      using context::Manifold;
      using ContinuousDynamicsBase = ContinuousDynamicsAbstractTpl<Scalar>;
      using ContinuousDynamicsData = ContinuousDynamicsDataTpl<Scalar>;
      using ODEAbstract = ODEAbstractTpl<Scalar>;
      using ODEData = ODEDataTpl<Scalar>;

      bp::class_<internal::PyContinuousDynamics<>>(
        "ContinuousDynamicsBase",
        "Base class for continuous-time dynamical models (DAEs and ODEs).",
        bp::init<const Manifold&, const int>(
          "Default constructor: provide the working manifold and control space dimension.",
          bp::args("self", "space", "nu"))
      )
        .def("evaluate",
             bp::pure_virtual(&ContinuousDynamicsBase::evaluate),
             bp::args("self", "x", "u", "xdot", "data"),
             "Evaluate the DAE functions.")
        .def("computeJacobians",  
             bp::pure_virtual(&ContinuousDynamicsBase::computeJacobians),
             bp::args("self", "x", "u", "xdot", "data"),
             "Evaluate the DAE function derivatives.")
        .def(CreateDataPythonVisitor<ContinuousDynamicsBase>());

      bp::register_ptr_to_python<shared_ptr<ContinuousDynamicsData>>();
      bp::class_<ContinuousDynamicsData>(
        "ContinuousDynamicsData",
        "Data struct for continuous dynamics/DAE models.",
        bp::init<int, int>()
      )
        .def_readwrite("value", &ContinuousDynamicsData::value_, "Vector value of the DAE residual.")
        .def_readwrite("Jx", &ContinuousDynamicsData::Jx_, "Jacobian with respect to state.")
        .def_readwrite("Ju", &ContinuousDynamicsData::Ju_, "Jacobian with respect to controls.")
        .def_readwrite("Jxdot", &ContinuousDynamicsData::Jxdot_, "Jacobian with respect to :math:`\\dot{x}`.")
        ;
      
      /* ODEs */

      bp::class_<internal::PyODEBase, bp::bases<ContinuousDynamicsBase>>(
        "ODEAbstract", "Continuous dynamics described by ordinary differential equations (ODEs).",
        bp::init<const Manifold&, const int>(bp::args("self", "space", "nu"))
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
        "ODEData", "Data struct for ODE models.", bp::init<int, int>()
      )
        .def_readwrite("xdot", &ODEData::xdot_);

    }

  } // namespace python 
} // namespace proxddp
