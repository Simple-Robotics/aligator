
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/dynamics.hpp"


namespace proxddp
{
  namespace python
  {
    
    void exposeDynamics()
    {
      using context::Scalar;
      using namespace proxddp::dynamics;

      using ContinuousDynamicsBase = ContinuousDynamicsTpl<Scalar>;

      bp::class_<
        ContinuousDynamicsBase,
        internal::PyContinuousDynamics<>
        >
      (
        "ContinuousDynamicsBase", "Base class for continuous dynamics/DAE models.",
        bp::init<const context::Manifold&, const int>(
          bp::args("self", "space", "nu")
          )
      )
        .def("evaluate",   bp::pure_virtual(&ContinuousDynamicsBase::evaluate),
             bp::args("self", "x", "u", "xdot", "data"))
        .def("computeJacobians",  bp::pure_virtual(&ContinuousDynamicsBase::computeJacobians),
             bp::args("self", "x", "u", "xdot", "data"))
        .def(CreateDataPythonVisitor<ContinuousDynamicsBase>());

      using ContData = dynamics::ContinuousDynamicsDataTpl<Scalar>;
      bp::register_ptr_to_python<shared_ptr<ContData>>();
      bp::class_<ContData>
      (
        "ContinuousDynamicsData", "Data struct for continuous dynamics/DAE models.",
        bp::no_init
      )
        .def_readwrite("value", &ContData::error_)
        .def_readwrite("Jx", &ContData::Jx_)
        .def_readwrite("Ju", &ContData::Ju_)
        .def_readwrite("Jxdot", &ContData::Jxdot_)
        ;
      
      using ODEBase = ODEBaseTpl<Scalar>;
      using ODEData = ODEDataTpl<Scalar>;
      bp::register_ptr_to_python<shared_ptr<ODEData>>();
      bp::class_<ODEBase,
                 bp::bases<ContinuousDynamicsBase>,
                 internal::PyODEBase
                 >
      (
        "ODEBase", "Continuous dynamics described by ordinary differential equations (ODEs).",
        bp::init<const context::Manifold&, const int>(
          bp::args("self", "space", "nu"))
      )
        .def("forward",  bp::pure_virtual(&ODEBase::forward),
             bp::args("self", "x", "u", "xdot_out"),
             "Compute the value of the ODE vector field."
             )
        .def("dForward", bp::pure_virtual(&ODEBase::dForward),
             bp::args("self", "x", "u", "Jout_x", "Jout_u"),
             "Compute the derivatives of the ODE vector field wrt (x, u)."
             )
        .def(CreateDataPythonVisitor<ODEBase>());

      bp::class_<ODEData,
                 bp::bases<ContData>
                 >
      (
        "ODEData", "Data struct for ODE models.", bp::no_init
      )
        .def_readwrite("xdot", &ODEData::xdot_)
      ;

    }
    
  } // namespace python
} // namespace proxddp


