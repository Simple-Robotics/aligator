/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/core/common-model-builder-container.hpp"

namespace aligator {
namespace python {
namespace internal {
/// @brief Wrapper for the CostDataAbstractTpl class and its children.
template <typename T = context::CostAbstract>
struct PyCostFunction : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  using bp::wrapper<T>::get_override;
  using CostData = CostDataAbstractTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  /// forwarding constructor
  template <typename... Args>
  PyCostFunction(Args &&...args) : T(std::forward<Args>(args)...) {}

  virtual void
  configure(CommonModelBuilderContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, T, configure, container);
  }

  void default_configure(CommonModelBuilderContainer &container) const {
    T::configure(container);
  }

  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, boost::ref(data));
  }

  virtual void computeGradients(const ConstVectorRef &x,
                                const ConstVectorRef &u,
                                CostData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeGradients", x, u,
                                  boost::ref(data));
  }

  virtual void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                               CostData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeHessians", x, u,
                                  boost::ref(data));
  }

  virtual shared_ptr<CostData>
  createData(const CommonModelDataContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<CostData>, T, createData, container);
  }

  shared_ptr<CostData>
  default_createData(const CommonModelDataContainer &container) const {
    return T::createData(container);
  }
};

} // namespace internal

} // namespace python
} // namespace aligator
