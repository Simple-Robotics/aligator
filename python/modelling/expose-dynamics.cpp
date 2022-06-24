
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/dynamics.hpp"

#include "proxddp/modelling/linear-discrete-dynamics.hpp"


namespace proxddp
{
  namespace python
  {
    
    void exposeODEs()
    {
      using namespace proxddp::dynamics;
      using context::Scalar;
      using ContinuousDynamicsBase = ContinuousDynamicsAbstractTpl<Scalar>;
      using ContinuousDynamicsData = ContinuousDynamicsDataTpl<Scalar>;
      using ODEBase = ODEAbstractTpl<Scalar>;
      using ODEData = ODEDataTpl<Scalar>;

      bp::class_<internal::PyContinuousDynamics<>>(
        "ContinuousDynamicsBase",
        "Base class for continuous-time dynamical models (DAEs and ODEs).",
        bp::init<const context::Manifold&, const int>(
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
        bp::no_init
      )
        .def_readwrite("value", &ContinuousDynamicsData::value_, "Vector value of the DAE residual.")
        .def_readwrite("Jx", &ContinuousDynamicsData::Jx_, "Jacobian with respect to state.")
        .def_readwrite("Ju", &ContinuousDynamicsData::Ju_, "Jacobian with respect to controls.")
        .def_readwrite("Jxdot", &ContinuousDynamicsData::Jxdot_, "Jacobian with respect to :math:`\\dot{x}`.")
        ;
      
      /* ODEs */

      bp::class_<internal::PyODEBase, bp::bases<ContinuousDynamicsBase>>(
        "ODEBase", "Continuous dynamics described by ordinary differential equations (ODEs).",
        bp::init<const context::Manifold&, const int>(bp::args("self", "space", "nu"))
      )
        .def("forward",  bp::pure_virtual(&ODEBase::forward),
             bp::args("self", "x", "u", "data"),
             "Compute the value of the ODE vector field, i.e. the "
             "state time derivative :math:`\\dot{x}`.")
        .def("dForward", bp::pure_virtual(&ODEBase::dForward),
             bp::args("self", "x", "u", "data"),
             "Compute the derivatives of the ODE vector field with respect "
             "to the state-control pair :math:`(x, u)`.")
        .def(CreateDataPythonVisitor<ODEBase>());

      bp::register_ptr_to_python<shared_ptr<ODEData>>();

      bp::class_<ODEData, bp::bases<ContinuousDynamicsData>>(
        "ODEData", "Data struct for ODE models.", bp::no_init
      )
        .def_readwrite("xdot", &ODEData::xdot_);


    }

    void exposeDynamics()
    {
      using context::Scalar;
      using context::Manifold;
      using context::DynamicsModel;
      using context::MatrixXs;
      using context::VectorXs;

      using namespace proxddp::dynamics;

      using ManifoldPtr = shared_ptr<context::Manifold>;
      using context::ExplicitDynamics;
      bp::class_<internal::PyExplicitDynamicsModel,
                 bp::bases<DynamicsModel>,
                 boost::noncopyable>
      (
        "ExplicitDynamicsModel", "Base class for explicit dynamics.",
        bp::init<const int, const int, const ManifoldPtr&>(
          bp::args("self", "ndx1", "nu", "next_space")
        )
      )
        .def(bp::init<const ManifoldPtr&, const int>(
          bp::args("self", "space", "nu")))
        .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
              bp::args("self", "x", "u", "data"),
              "Call for forward discrete dynamics.")
        .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
             bp::args("self", "x", "u", "data"),
             "Compute the derivatives of forward discrete dynamics.")
        .add_property("space", bp::make_function(&ExplicitDynamics::out_space, bp::return_internal_reference<>()),
                      "Output space.")
        .def(CreateDataPythonVisitor<ExplicitDynamics>());

      bp::register_ptr_to_python<shared_ptr<context::ExplicitDynData>>();

      bp::class_<context::ExplicitDynData, bp::bases<context::StageFunctionData>>(
        "ExplicitDynamicsData", "Data struct for explicit dynamics models.", bp::no_init)
        .add_property("dx",   bp::make_getter(&context::ExplicitDynData::dx_, bp::return_value_policy<bp::return_by_value>()))
        .add_property("xout", bp::make_getter(&context::ExplicitDynData::xoutref_, bp::return_value_policy<bp::return_by_value>()));

      /* Expose implementations */

      bp::class_<LinearDiscreteDynamicsTpl<Scalar>, bp::bases<context::ExplicitDynamics>>(
        "LinearDiscreteDynamics",
        "Linear discrete dynamics x[t+1] = Ax[t] + Bu[t] in Euclidean space, or on the tangent state space.",
        bp::init<const MatrixXs&, const MatrixXs&, const VectorXs&>(
          (bp::arg("self"), bp::arg("A"), bp::arg("B"), bp::arg("c"))
        )
      )
        .def_readonly("A", &LinearDiscreteDynamicsTpl<Scalar>::A_)
        .def_readonly("B", &LinearDiscreteDynamicsTpl<Scalar>::B_)
        .def_readonly("c", &LinearDiscreteDynamicsTpl<Scalar>::c_)
        ;

    }
    
  } // namespace python
} // namespace proxddp


