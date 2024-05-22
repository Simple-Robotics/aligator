/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/python/functions.hpp"

#include "aligator/modelling/function-xpr-slice.hpp"

namespace aligator {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;
using internal::PyUnaryFunction;

/// Expose the UnaryFunction type and its member function overloads.
void exposeUnaryFunctions() {
  using unary_eval_t = void (UnaryFunction::*)(const ConstVectorRef &,
                                               StageFunctionData &) const;
  using full_eval_t =
      void (UnaryFunction::*)(const ConstVectorRef &, const ConstVectorRef &,
                              const ConstVectorRef &, StageFunctionData &)
          const;
  using unary_vhp_t =
      void (UnaryFunction::*)(const ConstVectorRef &, const ConstVectorRef &,
                              StageFunctionData &) const;
  using full_vhp_t = void (UnaryFunction::*)(
      const ConstVectorRef &, const ConstVectorRef &, const ConstVectorRef &,
      const ConstVectorRef &, StageFunctionData &) const;
  bp::register_ptr_to_python<shared_ptr<UnaryFunction>>();
  bp::class_<PyUnaryFunction<>, bp::bases<StageFunction>, boost::noncopyable>(
      "UnaryFunction",
      "Base class for unary functions of the form :math:`x \\mapsto f(x)`.",
      bp::no_init)
      .def(bp::init<const int, const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr")))
      .def("evaluate", bp::pure_virtual<unary_eval_t>(&UnaryFunction::evaluate),
           bp::args("self", "x", "data"))
      .def<full_eval_t>("evaluate", &UnaryFunction::evaluate,
                        bp::args("self", "x", "u", "y", "data"))
      .def("computeJacobians",
           bp::pure_virtual<unary_eval_t>(&UnaryFunction::computeJacobians),
           bp::args("self", "x", "data"))
      .def<full_eval_t>("computeJacobians", &UnaryFunction::computeJacobians,
                        bp::args("self", "x", "u", "y", "data"))
      .def("computeVectorHessianProducts",
           bp::pure_virtual<unary_vhp_t>(
               &UnaryFunction::computeVectorHessianProducts),
           bp::args("self", "x", "lbda", "data"))
      .def<full_vhp_t>("computeVectorHessianProducts",
                       &UnaryFunction::computeVectorHessianProducts,
                       bp::args("self", "x", "u", "y", "lbda", "data"))
      .def(SlicingVisitor<UnaryFunction>());
}

} // namespace python
} // namespace aligator
