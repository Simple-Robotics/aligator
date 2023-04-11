/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {
namespace python {
namespace internal {
/// Wrapper for the StageFunction class and any virtual children that avoids
/// having to redeclare Python overrides for these children.
///
/// This implements the "trampoline" technique from Pybind11's docs:
/// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance
///
/// @tparam FunctionBase The virtual class to expose.
template <class FunctionBase = context::StageFunction>
struct PyStageFunction : FunctionBase, bp::wrapper<FunctionBase> {
  using Scalar = typename FunctionBase::Scalar;
  using Data = FunctionDataTpl<Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  // Use perfect forwarding to the FunctionBase constructors.
  template <typename... Args>
  PyStageFunction(Args &&...args) : FunctionBase(std::forward<Args>(args)...) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, y, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, y, data);
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
                                    const ConstVectorRef &lbda,
                                    Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE(void, FunctionBase, computeVectorHessianProducts, x,
                            u, y, lbda, data);
  }

  shared_ptr<Data> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<Data>, FunctionBase, createData, );
  }

  shared_ptr<Data> default_createData() const {
    return FunctionBase::createData();
  }
};

} // namespace internal
} // namespace python
} // namespace proxddp
