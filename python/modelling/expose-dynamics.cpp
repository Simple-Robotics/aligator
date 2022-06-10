
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/dynamics.hpp"

#include "proxddp/modelling/linear-discrete-dynamics.hpp"


namespace proxddp
{
  namespace python
  {
    
    void exposeDynamics()
    {
      using context::Scalar;
      using context::Manifold;
      using context::DynamicsModel;
      using namespace proxddp::dynamics;

      using ManifoldPtr = shared_ptr<context::Manifold>;
      using context::ExplicitDynamics;
      bp::class_<internal::PyExplicitDynamicsModel,
                 bp::bases<DynamicsModel>,
                 boost::noncopyable>
      (
        "ExplicitDynamicsModel", "Base class for explicit dynamics.",
        bp::init<const int, const int, const ManifoldPtr&>(
          bp::args("self", "ndx1", "nu", "out_space")
        )
      )
        .def(bp::init<const ManifoldPtr&, const int>(
          bp::args("self", "out_space", "nu")
        ))
        .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
              bp::args("self", "x", "u", "out"),
              "Call for forward discrete dynamics.")
        .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
              bp::args("self", "x", "u", "Jx", "Ju"),
              "Compute the derivatives of forward discrete dynamics.")
        .def_readonly("space", &ExplicitDynamics::out_space_, "Output space.")
        .def(CreateDataPythonVisitor<ExplicitDynamics>());

      bp::class_<context::ExplicitDynData,
                 bp::bases<context::FunctionData>,
                 boost::noncopyable>(
                   "ExplicitDynamicsData",
                   "Data struct for explicit dynamics models.",
                   bp::no_init
                 )
        .def_readwrite("xout", &context::ExplicitDynData::xout_);

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

      using ContinuousDynamicsData = dynamics::ContinuousDynamicsDataTpl<Scalar>;
      bp::register_ptr_to_python<shared_ptr<ContinuousDynamicsData>>();
      bp::class_<ContinuousDynamicsData>
      (
        "ContinuousDynamicsData", "Data struct for continuous dynamics/DAE models.",
        bp::no_init
      )
        .def_readwrite("value", &ContinuousDynamicsData::error_)
        .def_readwrite("Jx", &ContinuousDynamicsData::Jx_)
        .def_readwrite("Ju", &ContinuousDynamicsData::Ju_)
        .def_readwrite("Jxdot", &ContinuousDynamicsData::Jxdot_)
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
                 bp::bases<ContinuousDynamicsData>
                 >
      (
        "ODEData", "Data struct for ODE models.", bp::no_init
      )
        .def_readwrite("xdot", &ODEData::xdot_)
      ;

      using context::MatrixXs;
      using context::VectorXs;

      bp::class_<LinearDiscreteDynamics<Scalar>, bp::bases<context::ExplicitDynamics>>(
        "LinearDiscreteDynamics",
        "Linear discrete dynamics x[t+1] = Ax[t] + Bu[t] in Euclidean space, or on the tangent state space.",
        bp::init<const MatrixXs&, const MatrixXs&, const VectorXs&>(
          (bp::arg("self"), bp::arg("A"), bp::arg("B"), bp::arg("c"))
        )
      )
        .def_readonly("A", &LinearDiscreteDynamics<Scalar>::A_)
        .def_readonly("B", &LinearDiscreteDynamics<Scalar>::B_)
        .def_readonly("c", &LinearDiscreteDynamics<Scalar>::c_)
        ;

    }
    
  } // namespace python
} // namespace proxddp


