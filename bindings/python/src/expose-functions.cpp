/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/python/functions.hpp"
#include "aligator/python/eigen-member.hpp"

#include "aligator/modelling/state-error.hpp"
#include "aligator/modelling/linear-function.hpp"
#include "aligator/modelling/control-box-function.hpp"
#include "aligator/modelling/function-xpr-slice.hpp"

namespace aligator {
namespace python {

using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::DynamicsModel;
using context::MatrixXs;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;
using context::VectorXs;
using PolyFunction = xyz::polymorphic<StageFunction>;
using StateErrorResidual = StateErrorResidualTpl<Scalar>;
using ControlErrorResidual = ControlErrorResidualTpl<Scalar>;
PolymorphicMultiBaseVisitor<StageFunction> func_visitor;

/// Required trampoline class
struct FunctionDataWrapper : StageFunctionData, bp::wrapper<StageFunctionData> {
  using StageFunctionData::StageFunctionData;
};

void exposeFunctionBase() {
  register_polymorphic_to_python<PolyFunction>();
  bp::class_<PyStageFunction<>, boost::noncopyable>(
      "StageFunction",
      "Base class for ternary functions f(x,u,x') on a stage of the problem.",
      bp::no_init)
      .def(bp::init<const int, const int, const int>(
          ("self"_a, "ndx", "nu", "nr")))
      .def("evaluate", bp::pure_virtual(&StageFunction::evaluate),
           ("self"_a, "x", "u", "data"))
      .def("computeJacobians",
           bp::pure_virtual(&StageFunction::computeJacobians),
           ("self"_a, "x", "u", "data"))
      .def("computeVectorHessianProducts",
           &StageFunction::computeVectorHessianProducts,
           ("self"_a, "x", "u", "lbda", "data"))
      .def_readonly("ndx1", &StageFunction::ndx1, "Current state space.")
      .def_readonly("nu", &StageFunction::nu, "Control dimension.")
      .def_readonly("nr", &StageFunction::nr, "Function codimension.")
      .def(SlicingVisitor<StageFunction>())
      .def(func_visitor)
      .def(CreateDataPolymorphicPythonVisitor<StageFunction,
                                              PyStageFunction<>>())
      .enable_pickling_(true);

  bp::register_ptr_to_python<shared_ptr<StageFunctionData>>();

  bp::class_<FunctionDataWrapper, boost::noncopyable>(
      "StageFunctionData", "Data struct for holding data about functions.",
      bp::no_init)
      .def(bp::init<const StageFunction &>(("self"_a, "model")))
      .def(bp::init<int, int, int>(bp::args("self", "ndx", "nu", "nr")))
      .add_property(
          "value",
          bp::make_getter(&StageFunctionData::valref_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Function value.")
      .add_property("jac_buffer",
                    make_getter_eigen_matrix(&StageFunctionData::jac_buffer_),
                    "Buffer of the full function Jacobian wrt (x,u,y).")
      .add_property(
          "vhp_buffer",
          make_getter_eigen_matrix(&StageFunctionData::vhp_buffer_),
          "Buffer of the full function vector-Hessian product wrt (x,u,y).")
      .add_property(
          "Jx",
          bp::make_getter(&StageFunctionData::Jx_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $x$.")
      .add_property(
          "Ju",
          bp::make_getter(&StageFunctionData::Ju_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $u$.")
      .add_property(
          "Hxx",
          bp::make_getter(&StageFunctionData::Hxx_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, x)$.")
      .add_property(
          "Hxu",
          bp::make_getter(&StageFunctionData::Hxu_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, u)$.")
      .add_property(
          "Huu",
          bp::make_getter(&StageFunctionData::Huu_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(u, u)$.")
      .def(PrintableVisitor<StageFunctionData>())
      .def(PrintAddressVisitor<StageFunctionData>());

  StdVectorPythonVisitor<std::vector<PolyFunction>, true>::expose(
      "StdVec_StageFunction",
      eigenpy::details::overload_base_get_item_for_std_vector<
          std::vector<PolyFunction>>{});
  StdVectorPythonVisitor<std::vector<shared_ptr<StageFunctionData>>,
                         true>::expose("StdVec_StageFunctionData");
}

// fwd-decl exposeUnaryFunctions()
void exposeUnaryFunctions();

// fwd-decl exposeFunctionExpressions()
void exposeFunctionExpressions();

void exposeFunctions() {
  exposeFunctionBase();
  exposeUnaryFunctions();
  exposeFunctionExpressions();

  bp::class_<StateErrorResidual, bp::bases<UnaryFunction>>(
      "StateErrorResidual",
      bp::init<const xyz::polymorphic<context::Manifold> &, const int,
               const context::VectorXs &>(("self"_a, "space", "nu", "target")))
      .def_readonly("space", &StateErrorResidual::space_)
      .def_readwrite("target", &StateErrorResidual::target_)
      .def(PolymorphicMultiBaseVisitor<UnaryFunction, StageFunction>());

  bp::class_<ControlErrorResidual, bp::bases<StageFunction>>(
      "ControlErrorResidual",
      bp::init<const int, const xyz::polymorphic<context::Manifold> &,
               const context::VectorXs &>(
          ("self"_a, "ndx", "uspace", "target")))
      .def(bp::init<const int, const context::VectorXs &>(
          ("self"_a, "ndx", "target")))
      .def(bp::init<int, int>(("self"_a, "ndx", "nu")))
      .def_readonly("space", &ControlErrorResidual::space_)
      .def_readwrite("target", &ControlErrorResidual::target_)
      .def(func_visitor);

  using LinearFunction = LinearFunctionTpl<Scalar>;
  bp::class_<LinearFunction, bp::bases<StageFunction>>(
      "LinearFunction",
      bp::init<const int, const int, const int>(("self"_a, "ndx", "nu", "nr")))
      .def(bp::init<const ConstMatrixRef &, const ConstMatrixRef &,
                    const ConstVectorRef &>("Constructor with C=0.",
                                            ("self"_a, "A", "B", "d")))
      .def_readonly("A", &LinearFunction::A_)
      .def_readonly("B", &LinearFunction::B_)
      .def_readonly("d", &LinearFunction::d_)
      .def(func_visitor);

  bp::class_<ControlBoxFunctionTpl<Scalar>, bp::bases<StageFunction>>(
      "ControlBoxFunction",
      bp::init<const int, const VectorXs &, const VectorXs &>(
          bp::args("self", "ndx", "umin", "umax")))
      .def(bp::init<const int, const int, const Scalar, const Scalar>(
          bp::args("self", "ndx", "nu", "umin", "umax")))
      .def(func_visitor);
}

} // namespace python
} // namespace aligator
