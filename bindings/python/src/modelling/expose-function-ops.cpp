/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "aligator/python/fwd.hpp"
#include "aligator/python/eigen-member.hpp"
#include "aligator/modelling/function-xpr-slice.hpp"
#include "aligator/modelling/linear-function-composition.hpp"

namespace aligator {
namespace python {

using context::MatrixXs;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;
using context::VectorXs;

template <typename Base> void exposeSliceExpression(const char *name) {
  // FUNCTION SLICE

  using FunctionSliceXpr = FunctionSliceXprTpl<Scalar, Base>;

  bp::register_ptr_to_python<xyz::polymorphic<FunctionSliceXpr>>();
  bp::class_<FunctionSliceXpr, bp::bases<Base>>(
      name,
      "Represents a slice of an expression according to either a single index "
      "or an array of indices.",
      bp::init<xyz::polymorphic<Base>, std::vector<int> const &>(
          bp::args("self", "func", "indices")))
      .def(bp::init<xyz::polymorphic<Base>, const int>(
          "Constructor from a single index.", bp::args("self", "func", "idx")))
      .def_readonly("func", &FunctionSliceXpr::func, "Underlying function.")
      .def_readonly("indices", &FunctionSliceXpr::indices,
                    "Indices of the slice.");
}

template <typename LFC>
struct LinFunctionCompositionVisitor
    : bp::def_visitor<LinFunctionCompositionVisitor<LFC>> {
  using FunType = typename LFC::Base;

  template <class PyClass> void visit(PyClass &cl) const {
    cl.def(bp::init<xyz::polymorphic<FunType>, const MatrixXs, const VectorXs>(
               "Construct a composition from the underlying function, weight "
               "matrix "
               ":math:`A` and bias :math:`b`.",
               bp::args("self", "func", "A", "b")))
        .def(bp::init<xyz::polymorphic<FunType>, const context::MatrixXs>(
            "Constructor where the bias :math:`b` is assumed to be zero.",
            bp::args("self", "func", "A")))
        .def_readonly("func", &LFC::func, "The underlying function.")
        .def_readonly("A", &LFC::A, "Weight matrix.")
        .def_readonly("b", &LFC::b, "Bias vector.");
    bp::class_<typename LFC::Data, bp::bases<StageFunctionData>,
               boost::noncopyable>("LFC", bp::no_init)
        .def_readonly("sub_data", &LFC::Data::sub_data);

    bp::def(
        "linear_compose",
        +[](xyz::polymorphic<FunType> func, const MatrixXs &A,
            const VectorXs &b) { return linear_compose(func, A, b); },
        bp::args("func", "A", "b"));
  }
};

void exposeFunctionExpressions() {

  exposeSliceExpression<StageFunction>("StageFunctionSliceXpr");
  exposeSliceExpression<UnaryFunction>("UnaryFunctionSliceXpr");

  using FunctionSliceData = FunctionSliceDataTpl<Scalar>;
  bp::class_<FunctionSliceData, bp::bases<StageFunctionData>,
             boost::noncopyable>("FunctionSliceData", bp::no_init)
      .def_readonly("sub_data", &FunctionSliceData::sub_data,
                    "Underlying function's data.");

  /// FUNCTION LINEAR COMPOSE

  using LinearFunctionComposition = LinearFunctionCompositionTpl<Scalar>;
  using LinearUnaryFunctionComposition =
      LinearUnaryFunctionCompositionTpl<Scalar>;

  bp::register_ptr_to_python<xyz::polymorphic<LinearFunctionComposition>>();
  bp::class_<LinearFunctionComposition, bp::bases<StageFunction>>(
      "LinearFunctionComposition",
      "Function composition :math:`r(x) = Af(x, u, y) + b`.", bp::no_init)
      .def(LinFunctionCompositionVisitor<LinearFunctionComposition>());

  bp::register_ptr_to_python<
      xyz::polymorphic<LinearUnaryFunctionComposition>>();
  bp::class_<LinearUnaryFunctionComposition, bp::bases<UnaryFunction>>(
      "LinearUnaryFunctionComposition",
      "Function composition for unary functions: :math:`r(x) = Af(x) + b`.",
      bp::no_init)
      .def(LinFunctionCompositionVisitor<LinearUnaryFunctionComposition>());
}

} // namespace python
} // namespace aligator
