/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/functions.hpp"
#include "proxddp/python/eigen-member.hpp"

#include "proxddp/modelling/state-error.hpp"
#include "proxddp/modelling/linear-function.hpp"
#include "proxddp/modelling/control-box-function.hpp"
#include "proxddp/modelling/function-xpr-slice.hpp"
#include "proxddp/modelling/linear-function-composition.hpp"

namespace proxddp {
namespace python {

void exposeFunctionBase() {
  using context::ConstMatrixRef;
  using context::ConstVectorRef;
  using context::DynamicsModel;
  using context::FunctionData;
  using context::MatrixXs;
  using context::Scalar;
  using context::StageFunction;
  using context::VectorXs;
  using internal::PyStageFunction;
  using FunctionPtr = shared_ptr<StageFunction>;

  bp::register_ptr_to_python<FunctionPtr>();

  bp::class_<PyStageFunction<>, boost::noncopyable>(
      "StageFunction",
      "Base class for ternary functions f(x,u,x') on a stage of the problem.",
      bp::init<const int, const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr")))
      .def(bp::init<const int, const int, const int>(
          bp::args("self", "ndx", "nu", "nr")))
      .def("evaluate", bp::pure_virtual(&StageFunction::evaluate),
           bp::args("self", "x", "u", "y", "data"))
      .def("computeJacobians",
           bp::pure_virtual(&StageFunction::computeJacobians),
           bp::args("self", "x", "u", "y", "data"))
      .def("computeVectorHessianProducts",
           &StageFunction::computeVectorHessianProducts,
           bp::args("self", "x", "u", "y", "lbda", "data"))
      .def_readonly("ndx1", &StageFunction::ndx1, "Current state space.")
      .def_readonly("ndx2", &StageFunction::ndx2, "Next state space.")
      .def_readonly("nu", &StageFunction::nu, "Control dimension.")
      .def_readonly("nr", &StageFunction::nr, "Function codimension.")
      .def(CreateDataPythonVisitor<StageFunction>());

  bp::register_ptr_to_python<shared_ptr<FunctionData>>();

  bp::class_<FunctionData>("FunctionData",
                           "Data struct for holding data about functions.",
                           bp::init<int, int, int, int>(
                               bp::args("self", "ndx1", "nu", "ndx2", "nr")))
      .add_property(
          "value",
          bp::make_getter(&FunctionData::valref_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Function value.")
      .add_property("jac_buffer_",
                    make_getter_eigen_matrix(&FunctionData::jac_buffer_),
                    "Buffer of the full function Jacobian wrt (x,u,y).")
      .add_property(
          "vhp_buffer", make_getter_eigen_matrix(&FunctionData::vhp_buffer_),
          "Buffer of the full function vector-Hessian product wrt (x,u,y).")
      .add_property(
          "Jx",
          bp::make_getter(&FunctionData::Jx_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $x$.")
      .add_property(
          "Ju",
          bp::make_getter(&FunctionData::Ju_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $u$.")
      .add_property(
          "Jy",
          bp::make_getter(&FunctionData::Jy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $y$.")
      .add_property(
          "Hxx",
          bp::make_getter(&FunctionData::Hxx_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, x)$.")
      .add_property(
          "Hxu",
          bp::make_getter(&FunctionData::Hxu_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, u)$.")
      .add_property(
          "Hxy",
          bp::make_getter(&FunctionData::Hxy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, y)$.")
      .add_property(
          "Huu",
          bp::make_getter(&FunctionData::Huu_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(u, u)$.")
      .add_property(
          "Huy",
          bp::make_getter(&FunctionData::Huy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, y)$.")
      .add_property(
          "Hyy",
          bp::make_getter(&FunctionData::Hyy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(y, y)$.")
      .def(PrintableVisitor<FunctionData>())
      .def(ClonePythonVisitor<FunctionData>());

  StdVectorPythonVisitor<std::vector<FunctionPtr>, true>::expose(
      "StdVec_StageFunction", "Vector of function objects.");
  StdVectorPythonVisitor<std::vector<shared_ptr<FunctionData>>, true>::expose(
      "StdVec_FunctionData", "Vector of function data objects.");

  bp::class_<StateErrorResidualTpl<Scalar>, bp::bases<StageFunction>>(
      "StateErrorResidual", bp::init<const shared_ptr<context::Manifold> &,
                                     const int, const context::VectorXs &>(
                                bp::args("self", "space", "nu", "target")))
      .def_readonly("space", &StateErrorResidualTpl<Scalar>::space_)
      .def_readwrite("target", &StateErrorResidualTpl<Scalar>::target_);

  bp::class_<ControlErrorResidual<Scalar>, bp::bases<StageFunction>>(
      "ControlErrorResidual",
      bp::init<const int, const shared_ptr<context::Manifold> &,
               const context::VectorXs &>(
          bp::args("self", "ndx", "uspace", "target")))
      .def(bp::init<const int, const context::VectorXs &>(
          bp::args("self", "ndx", "target")))
      .def(bp::init<int, int>(bp::args("self", "ndx", "nu")))
      .def_readonly("space", &ControlErrorResidual<Scalar>::space_)
      .def_readwrite("target", &ControlErrorResidual<Scalar>::target_);

  bp::class_<LinearFunctionTpl<Scalar>, bp::bases<StageFunction>>(
      "LinearFunction", bp::init<const int, const int, const int, const int>(
                            bp::args("self", "ndx1", "nu", "ndx2", "nr")))
      .def(bp::init<const ConstMatrixRef, const ConstMatrixRef,
                    const ConstMatrixRef, const ConstVectorRef>(
          "Constructor with given matrices.",
          bp::args("self", "A", "B", "C", "d")))
      .def(bp::init<const ConstMatrixRef, const ConstMatrixRef,
                    const ConstVectorRef>("Constructor with C=0.",
                                          bp::args("self", "A", "B", "d")))
      .def_readonly("A", &LinearFunctionTpl<Scalar>::A_)
      .def_readonly("B", &LinearFunctionTpl<Scalar>::B_)
      .def_readonly("C", &LinearFunctionTpl<Scalar>::C_)
      .def_readonly("d", &LinearFunctionTpl<Scalar>::d_);

  bp::class_<ControlBoxFunctionTpl<Scalar>, bp::bases<StageFunction>>(
      "ControlBoxFunction",
      bp::init<const int, const VectorXs &, const VectorXs &>(
          bp::args("self", "ndx", "umin", "umax")))
      .def(bp::init<const int, const int, const Scalar, const Scalar>(
          bp::args("self", "ndx", "nu", "umin", "umax")));
}

void exposeFunctionExpressions() {
  using context::FunctionData;
  using context::Scalar;
  using context::StageFunction;

  using FunctionPtr = shared_ptr<StageFunction>;

  // FUNCTION SLICE

  using FunctionSliceXpr = FunctionSliceXprTpl<Scalar>;

  bp::class_<FunctionSliceXpr, bp::bases<StageFunction>>(
      "FunctionSliceXpr",
      "Represents a slice of an expression according to either a single index "
      "or an array of indices.",
      bp::init<FunctionPtr, std::vector<int>>(
          bp::args("self", "func", "indices")))
      .def(bp::init<FunctionPtr, const int>("Constructor from a single index.",
                                            bp::args("self", "func", "idx")))
      .def(CreateDataPythonVisitor<FunctionSliceXpr>());

  bp::class_<FunctionSliceXpr::OwnData, bp::bases<FunctionData>,
             boost::noncopyable>("FunctionSliceData", bp::no_init)
      .def_readonly("sub_data", &FunctionSliceXpr::OwnData::sub_data,
                    "Underlying function's data.");

  /// FUNCTION LINEAR COMPOSE

  using LinearFunctionComposition = LinearFunctionCompositionTpl<Scalar>;

  bp::class_<LinearFunctionComposition, bp::bases<StageFunction>>(
      "LinearFunctionComposition",
      "Function composition :math:`r(x) = Af(x) + b`.",
      bp::init<FunctionPtr, const context::MatrixXs, const context::VectorXs>(
          "Construct a composition from the underlying function, weight matrix "
          ":math:`A` and bias :math:`b`.",
          bp::args("self", "func", "A", "b")))
      .def(bp::init<FunctionPtr, const context::MatrixXs>(
          "Constructor where the bias :math:`b` is assumed to be zero.",
          bp::args("self", "func", "A")))
      .def_readonly("func", &LinearFunctionComposition::func,
                    "The underlying function.")
      .add_property("A",
                    make_getter_eigen_matrix(&LinearFunctionComposition::A),
                    "Weight matrix.")
      .add_property("b",
                    make_getter_eigen_matrix(&LinearFunctionComposition::A),
                    "Bias vector.");

  bp::class_<LinearFunctionComposition::OwnData, bp::bases<FunctionData>,
             boost::noncopyable>("LinearFunctionCompositionData", bp::no_init)
      .def_readonly("sub_data", &LinearFunctionComposition::OwnData::sub_data);
}

void exposeFunctions() {
  exposeFunctionBase();
  exposeFunctionExpressions();
}

} // namespace python
} // namespace proxddp
