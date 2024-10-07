/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/unary-function.hpp"

#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/integrator-abstract.hpp"

#include "aligator/python/polymorphic-convertible.hpp"

#include "proxsuite-nlp/python/polymorphic.hpp"

namespace aligator {
namespace python {
/// Wrapper for the StageFunction class and any virtual children that avoids
/// having to redeclare Python overrides for these children.
///
/// This implements the "trampoline" technique from Pybind11's docs:
/// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance
///
/// @tparam FunctionBase The virtual class to expose.
template <class FunctionBase = context::StageFunction>
struct PyStageFunction final
    : FunctionBase,
      proxsuite::nlp::python::PolymorphicWrapper<PyStageFunction<FunctionBase>,
                                                 FunctionBase> {
  using Scalar = typename FunctionBase::Scalar;
  using Data = StageFunctionDataTpl<Scalar>;
  using FunctionBase::FunctionBase;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, boost::ref(data));
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u,
                                  boost::ref(data));
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &lbda,
                                    Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, FunctionBase, computeVectorHessianProducts,
                             x, u, lbda, boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, FunctionBase, createData, );
  }

  shared_ptr<Data> default_createData() const {
    return FunctionBase::createData();
  }
};

template <typename UFunction = context::UnaryFunction>
struct PyUnaryFunction final
    : UFunction,
      proxsuite::nlp::python::PolymorphicWrapper<PyUnaryFunction<UFunction>,
                                                 UFunction> {
  using Scalar = typename UFunction::Scalar;
  static_assert(
      std::is_base_of_v<UnaryFunctionTpl<Scalar>, UFunction>,
      "Template parameter UFunction must derive from UnaryFunctionTpl<>.");
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using Data = StageFunctionDataTpl<Scalar>;

  using UFunction::UFunction;

  void evaluate(const ConstVectorRef &x, Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, boost::ref(data));
  }

  void computeJacobians(const ConstVectorRef &x, Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x,
                                  boost::ref(data));
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &lbda,
                                    Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, UFunction, computeVectorHessianProducts, x,
                             lbda, boost::ref(data));
  }

  void default_computeVectorHessianProducts(const ConstVectorRef &x,
                                            const ConstVectorRef &lbda,
                                            Data &data) const {
    UFunction::computeVectorHessianProducts(x, lbda, data);
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, UFunction, createData, );
  }

  shared_ptr<Data> default_createData() const {
    return UFunction::createData();
  }
};

template <typename Class>
struct SlicingVisitor : bp::def_visitor<SlicingVisitor<Class>> {
  using Scalar = typename Class::Scalar;
  using SliceType = FunctionSliceXprTpl<Scalar, Class>;

  template <typename Iterator, typename Fn>
  static auto do_with_slice(Fn &&fun, bp::slice::range<Iterator> &range) {
    while (range.start != range.stop) {
      fun(*range.start);
      std::advance(range.start, range.step);
    }
    fun(*range.start);
  }

  static auto get_slice(xyz::polymorphic<Class> const &fn,
                        bp::slice slice_obj) {
    std::vector<int> indices((unsigned)fn->nr);
    std::iota(indices.begin(), indices.end(), 0);
    auto bounds = slice_obj.get_indices(indices.cbegin(), indices.cend());
    std::vector<int> out{};

    do_with_slice([&](int i) { out.push_back(i); }, bounds);
    return SliceType(fn, out);
  }

  static auto get_from_index(xyz::polymorphic<Class> const &fn, const int idx) {
    return SliceType(fn, idx);
  }

  static auto get_from_indices(xyz::polymorphic<Class> const &fn,
                               std::vector<int> const &indices) {
    return SliceType(fn, indices);
  }

  template <typename... Args> void visit(bp::class_<Args...> &cl) const {
    cl.def("__getitem__", &get_from_index, bp::args("self", "idx"))
        .def("__getitem__", &get_from_indices, bp::args("self", "indices"))
        .def("__getitem__", &get_slice, bp::args("self", "sl"));
  }
};

} // namespace python
} // namespace aligator

namespace boost::python::objects {

template <>
struct value_holder<aligator::python::PyStageFunction<>>
    : proxsuite::nlp::python::OwningNonOwningHolder<
          aligator::python::PyStageFunction<>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

template <>
struct value_holder<aligator::python::PyUnaryFunction<>>
    : proxsuite::nlp::python::OwningNonOwningHolder<
          aligator::python::PyUnaryFunction<>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

} // namespace boost::python::objects
