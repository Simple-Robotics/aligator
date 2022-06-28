#include "proxddp/python/fwd.hpp"

#include "proxddp/python/functions.hpp"
#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/linear-function.hpp"
#include "proxddp/modelling/control-box-function.hpp"


namespace proxddp
{
  namespace python
  {

    void exposeFunctions()
    {
      using context::Scalar;
      using context::StageFunction;
      using context::DynamicsModel;
      using context::VectorXs;
      using context::MatrixXs;
      using context::ConstVectorRef;
      using context::ConstMatrixRef;

      bp::register_ptr_to_python<shared_ptr<StageFunction>>();

      bp::class_<internal::PyStageFunction<>, boost::noncopyable>(
        "StageFunction", "Base class for ternary functions f(x,u,x') on a stage of the problem.",
        bp::init<const int, const int, const int, const int>(bp::args("self", "ndx1", "nu", "ndx2", "nr"))
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
        .def_readonly("ndx1", &StageFunction::ndx1, "Current state space.")
        .def_readonly("ndx2", &StageFunction::ndx2, "Next state space.")
        .def_readonly("nu",   &StageFunction::nu, "Control dimension.")
        .def_readonly("nr",   &StageFunction::nr, "Function codimension.")
        .def(CreateDataPythonVisitor<StageFunction>());

      bp::register_ptr_to_python<shared_ptr<context::StageFunctionData>>();

      bp::class_<context::StageFunctionData>(
        "FunctionData", "Data struct for holding data about functions.",
        bp::init<const int, const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr")
        )
      )
        .add_property("value", bp::make_getter(&context::StageFunctionData::valref_, bp::return_value_policy<bp::return_by_value>()), "Function value.")
        .def_readonly("jac_buffer_", &context::StageFunctionData::jac_buffer_, "Buffer of the full function Jacobian wrt (x,u,y).")
        .def_readonly("vhp_buffer", &context::StageFunctionData::vhp_buffer_, "Buffer of the full function vector-Hessian product wrt (x,u,y).")
        .add_property("Jx",  bp::make_getter(&context::StageFunctionData::Jx_, bp::return_value_policy<bp::return_by_value>()), "Jacobian with respect to $x$.")
        .add_property("Ju",  bp::make_getter(&context::StageFunctionData::Ju_, bp::return_value_policy<bp::return_by_value>()), "Jacobian with respect to $u$.")
        .add_property("Jy",  bp::make_getter(&context::StageFunctionData::Jy_, bp::return_value_policy<bp::return_by_value>()), "Jacobian with respect to $y$.")
        .add_property("Hxx", bp::make_getter(&context::StageFunctionData::Hxx_, bp::return_value_policy<bp::return_by_value>()), "Hessian with respect to $(x, x)$.")
        .add_property("Hxu", bp::make_getter(&context::StageFunctionData::Hxu_, bp::return_value_policy<bp::return_by_value>()), "Hessian with respect to $(x, u)$.")
        .add_property("Hxy", bp::make_getter(&context::StageFunctionData::Hxy_, bp::return_value_policy<bp::return_by_value>()), "Hessian with respect to $(x, y)$.")
        .add_property("Huu", bp::make_getter(&context::StageFunctionData::Huu_, bp::return_value_policy<bp::return_by_value>()), "Hessian with respect to $(u, u)$.")
        .add_property("Huy", bp::make_getter(&context::StageFunctionData::Huy_, bp::return_value_policy<bp::return_by_value>()), "Hessian with respect to $(x, y)$.")
        .add_property("Hyy", bp::make_getter(&context::StageFunctionData::Hyy_, bp::return_value_policy<bp::return_by_value>()), "Hessian with respect to $(y, y)$.")
        .def(ClonePythonVisitor<context::StageFunctionData>());

      pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<StageFunction>>, true>::expose("StdVec_StageFunction", "Vector of function objects.");
      pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<context::StageFunctionData>>, true>::expose("StdVec_FunctionData", "Vector of function data objects.");

      bp::class_<StateErrorResidual<Scalar>, bp::bases<StageFunction>>(
        "StateErrorResidual",
        bp::init<const context::Manifold&, const int, const context::VectorXs&>(
          bp::args("self", "xspace", "nu", "target")
        )
      )
        .def_readwrite("target", &StateErrorResidual<Scalar>::target);

      bp::class_<ControlErrorResidual<Scalar>, bp::bases<StageFunction>>(
        "ControlErrorResidual",
        bp::init<const int, const context::Manifold&, const context::VectorXs&>(
          bp::args("self", "ndx", "uspace", "target"))
      )
        .def(bp::init<const int, const int, const context::VectorXs&>(bp::args("self", "ndx", "nu", "target")))
        .def_readwrite("target", &ControlErrorResidual<Scalar>::target);

      bp::class_<LinearFunctionTpl<Scalar>, bp::bases<StageFunction>>(
        "LinearFunction",
        bp::init<const int, const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr"))
      )
        .def(bp::init<const ConstMatrixRef, const ConstMatrixRef, const ConstMatrixRef, const ConstVectorRef>(
          "Constructor with given matrices.", bp::args("self", "A", "B", "C", "d")))
        .def(bp::init<const ConstMatrixRef, const ConstMatrixRef, const ConstVectorRef>(
          "Constructor with C=0.", bp::args("self", "A", "B", "d")))
        .def_readonly("A", &LinearFunctionTpl<Scalar>::A_)
        .def_readonly("B", &LinearFunctionTpl<Scalar>::B_)
        .def_readonly("C", &LinearFunctionTpl<Scalar>::C_)
        .def_readonly("d", &LinearFunctionTpl<Scalar>::d_);

      bp::class_<ControlBoxFunctionTpl<Scalar>, bp::bases<StageFunction>>(
        "ControlBoxFunction",
        bp::init<const int, const VectorXs, const VectorXs>(bp::args("self", "ndx", "umin", "umax"))
      )
        .def(bp::init<const int, const int, const Scalar, const Scalar>(
          bp::args("self", "ndx", "nu", "umin", "umax"))
          );


    }

  } // namespace python
} // namespace proxddp

