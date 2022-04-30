#include "proxddp/python/fwd.hpp"

#include "proxddp/core/node-function.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"


#include "proxddp/python/functions.hpp"


namespace proxddp
{
  namespace python
  {

    void exposeFunctions()
    {
      using context::Scalar;
      using context::DynamicsModel;
      using context::StageFunction;
      using internal::PyStageFunction;

      bp::class_<StageFunction, PyStageFunction<>, boost::noncopyable>(
        "StageFunction",
        "Base class for ternary functions f(x,u,x') on a stage of the problem.",
        bp::init<const int, const int, const int, const int>
                 (bp::args("self", "ndx1", "nu", "ndx2", "nr"))
      )
        .def(bp::init<const int, const int, const int>(bp::args("self", "ndx", "nu", "nr")))
        .def("evaluate",
             bp::pure_virtual(&StageFunction::evaluate),
             bp::args("self", "x", "u", "y", "data"))
        .def("computeJacobians",
             bp::pure_virtual(&StageFunction::computeJacobians),
             bp::args("self", "x", "u", "y", "data"))
        .def("computeVectorHessianProducts",
             &StageFunction::computeVectorHessianProducts,
             bp::args("self", "x", "u", "y", "lbda", "data"))
        .def("createData", &StageFunction::createData, "Create a data object.")
        .def_readonly("ndx1", &StageFunction::ndx1, "Current state space.")
        .def_readonly("ndx2", &StageFunction::ndx2, "Next state space.")
        .def_readonly("nu", &StageFunction::nu, "Control dimension.")
      ;

      bp::class_<context::FunctionData, shared_ptr<context::FunctionData>>(
        "FunctionData", "Data struct for holding data about functions.",
        bp::init<const int, const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr")
        )
      )
        .def_readonly("value", &context::FunctionData::value_,
                      "Function value.")
        .def_readonly("Jx", &context::FunctionData::Jx_,
                      "Jacobian with respect to $x$.")
        .def_readonly("Ju", &context::FunctionData::Ju_,
                      "Jacobian with respect to $u$.")
        .def_readonly("Jy", &context::FunctionData::Jy_,
                      "Jacobian with respect to $y$.")
        .def_readonly("Hxx", &context::FunctionData::Hxx_,
                      "Hessian with respect to $(x, x)$.")
        .def_readonly("Hxu", &context::FunctionData::Hxu_,
                      "Hessian with respect to $(x, u)$.")
        .def_readonly("Hxy", &context::FunctionData::Hxy_,
                      "Hessian with respect to $(x, y)$.")
        .def_readonly("Huu", &context::FunctionData::Huu_,
                      "Hessian with respect to $(u, u)$.")
        .def_readonly("Huy", &context::FunctionData::Huy_,
                      "Hessian with respect to $(x, y)$.")
        .def_readonly("Hyy", &context::FunctionData::Hyy_,
                      "Hessian with respect to $(y, y)$.")
      ;

      /** DYNAMICS **/

      bp::class_<DynamicsModel,
                 bp::bases<StageFunction>,
                 PyStageFunction<DynamicsModel>,
                 boost::noncopyable>(
        "DynamicsModel",
        "Dynamics models are specific ternary functions f(x,u,x') which map "
        "to the tangent bundle of the next state variable x'.",
        bp::init<const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2")
          )
      )
        .def(bp::init<const int, const int>(
          bp::args("self", "ndx", "nu")
          ))
      ;

      // bp::class_<internal::PyExplicitDynamicalModel,
      //            bp::bases<internal::PyDynWrap>, boost::noncopyable>
      // ("ExplicitDynamicsModel", "Explicit dynamics.", bp::no_init);

    }

  } // namespace python
} // namespace proxddp

