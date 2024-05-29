/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include <proxsuite-nlp/python/polymorphic.hpp>
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
using internal::PyStageFunction;
using PolyFunction = xyz::polymorphic<StageFunction>;
using StateErrorResidual = StateErrorResidualTpl<Scalar>;
using ControlErrorResidual = ControlErrorResidualTpl<Scalar>;

/// Required trampoline class
struct FunctionDataWrapper : StageFunctionData, bp::wrapper<StageFunctionData> {
  using StageFunctionData::StageFunctionData;
};

void exposeFunctionBase() {
  proxsuite::nlp::python::register_polymorphic_to_python<PolyFunction>();
  bp::class_<PyStageFunction<>, boost::noncopyable>(
      "StageFunction",
      "Base class for ternary functions f(x,u,x') on a stage of the problem.",
      bp::no_init)
      .def(bp::init<const int, const int, const int, const int>(
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
      //.def(SlicingVisitor<StageFunction>())
      .def(CreateDataPolymorphicPythonVisitor<StageFunction,
                                              PyStageFunction<>>());

  bp::register_ptr_to_python<shared_ptr<StageFunctionData>>();

  bp::class_<FunctionDataWrapper, boost::noncopyable>(
      "StageFunctionData", "Data struct for holding data about functions.",
      bp::init<int, int, int, int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr")))
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
          "Jy",
          bp::make_getter(&StageFunctionData::Jy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $y$.")
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
          "Hxy",
          bp::make_getter(&StageFunctionData::Hxy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, y)$.")
      .add_property(
          "Huu",
          bp::make_getter(&StageFunctionData::Huu_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(u, u)$.")
      .add_property(
          "Huy",
          bp::make_getter(&StageFunctionData::Huy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(x, y)$.")
      .add_property(
          "Hyy",
          bp::make_getter(&StageFunctionData::Hyy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Hessian with respect to $(y, y)$.")
      .def(PrintableVisitor<StageFunctionData>())
      .def(PrintAddressVisitor<StageFunctionData>())
      .def(ClonePythonVisitor<StageFunctionData>());

  StdVectorPythonVisitor<std::vector<PolyFunction>, true>::expose(
      "StdVec_StageFunction");
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

  bp::implicitly_convertible<StateErrorResidual,
                             xyz::polymorphic<UnaryFunction>>();
  bp::implicitly_convertible<StateErrorResidual,
                             xyz::polymorphic<StageFunction>>();
  bp::class_<StateErrorResidual, bp::bases<UnaryFunction>>(
      "StateErrorResidual",
      bp::init<const xyz::polymorphic<context::Manifold> &, const int,
               const context::VectorXs &>(
          bp::args("self", "space", "nu", "target")))
      .def_readonly("space", &StateErrorResidual::space_)
      .def_readwrite("target", &StateErrorResidual::target_);

  bp::implicitly_convertible<ControlErrorResidual,
                             xyz::polymorphic<StageFunction>>();
  bp::class_<ControlErrorResidual, bp::bases<StageFunction>>(
      "ControlErrorResidual",
      bp::init<const int, const xyz::polymorphic<context::Manifold> &,
               const context::VectorXs &>(
          bp::args("self", "ndx", "uspace", "target")))
      .def(bp::init<const int, const context::VectorXs &>(
          bp::args("self", "ndx", "target")))
      .def(bp::init<int, int>(bp::args("self", "ndx", "nu")))
      .def_readonly("space", &ControlErrorResidual::space_)
      .def_readwrite("target", &ControlErrorResidual::target_);

  using LinearFunction = LinearFunctionTpl<Scalar>;
  bp::implicitly_convertible<LinearFunction, xyz::polymorphic<StageFunction>>();
  bp::class_<LinearFunction, bp::bases<StageFunction>>(
      "LinearFunction", bp::init<const int, const int, const int, const int>(
                            bp::args("self", "ndx1", "nu", "ndx2", "nr")))
      .def(bp::init<const ConstMatrixRef, const ConstMatrixRef,
                    const ConstMatrixRef, const ConstVectorRef>(
          "Constructor with given matrices.",
          bp::args("self", "A", "B", "C", "d")))
      .def(bp::init<const ConstMatrixRef, const ConstMatrixRef,
                    const ConstVectorRef>("Constructor with C=0.",
                                          bp::args("self", "A", "B", "d")))
      .def_readonly("A", &LinearFunction::A_)
      .def_readonly("B", &LinearFunction::B_)
      .def_readonly("C", &LinearFunction::C_)
      .def_readonly("d", &LinearFunction::d_);

  bp::implicitly_convertible<ControlBoxFunctionTpl<Scalar>,
                             xyz::polymorphic<StageFunction>>();
  bp::class_<ControlBoxFunctionTpl<Scalar>, bp::bases<StageFunction>>(
      "ControlBoxFunction",
      bp::init<const int, const VectorXs &, const VectorXs &>(
          bp::args("self", "ndx", "umin", "umax")))
      .def(bp::init<const int, const int, const Scalar, const Scalar>(
          bp::args("self", "ndx", "nu", "umin", "umax")));
}

} // namespace python
} // namespace aligator
