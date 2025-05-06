/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/cost-abstract.hpp"

namespace aligator {
namespace python {
/// @brief Wrapper for the CostDataAbstractTpl class and its children.
struct PyCostFunction final
    : context::CostAbstract,
      PolymorphicWrapper<PyCostFunction, context::CostAbstract> {
  using Scalar = context::Scalar;
  using T = context::CostAbstract;
  using CostData = CostDataAbstractTpl<Scalar>;
  using context::CostAbstract::CostAbstractTpl;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, boost::ref(data));
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeGradients", x, u,
                                  boost::ref(data));
  }

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeHessians", x, u,
                                  boost::ref(data));
  }

  shared_ptr<CostData> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<CostData>, T, createData, );
  }

  shared_ptr<CostData> default_createData() const { return T::createData(); }
};
} // namespace python
} // namespace aligator

namespace boost::python::objects {

template <>
struct value_holder<aligator::python::PyCostFunction>
    : aligator::python::OwningNonOwningHolder<
          aligator::python::PyCostFunction> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

} // namespace boost::python::objects
