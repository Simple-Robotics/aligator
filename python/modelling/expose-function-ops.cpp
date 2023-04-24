/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxddp/python/fwd.hpp"
#include "proxddp/python/eigen-member.hpp"
#include "proxddp/modelling/function-xpr-slice.hpp"
#include "proxddp/modelling/linear-function-composition.hpp"

namespace proxddp {
namespace python {

using context::FunctionData;
using context::Scalar;
using context::StageFunction;
using FunctionPtr = shared_ptr<StageFunction>;

void exposeFunctionExpressions() {
  // FUNCTION SLICE

  using FunctionSliceXpr = FunctionSliceXprTpl<Scalar>;

  bp::class_<FunctionSliceXpr, bp::bases<StageFunction>>(
      "FunctionSliceXpr",
      "Represents a slice of an expression according to either a single index "
      "or an array of indices.",
      bp::init<FunctionPtr, std::vector<int>>(
          bp::args("self", "func", "indices")))
      .def(bp::init<FunctionPtr, const int>("Constructor from a single index.",
                                            bp::args("self", "func", "idx")));

  bp::class_<FunctionSliceXpr::Data, bp::bases<FunctionData>,
             boost::noncopyable>("FunctionSliceData", bp::no_init)
      .def_readonly("sub_data", &FunctionSliceXpr::Data::sub_data,
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

} // namespace python
} // namespace proxddp
