/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {
namespace python {
namespace internal {
/// Wrapper from StageFunction objects and their children that does
/// not require the child wrappers to create more virtual function overrides.
///
/// Using a templating technique from Pybind11's docs:
/// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance
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
};

} // namespace internal
} // namespace python
} // namespace proxddp
