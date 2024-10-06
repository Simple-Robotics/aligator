/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/core/dynamics.hpp"

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
template <class Base = context::DynamicsModel>
struct PyDynamics final
    : Base,
      proxsuite::nlp::python::PolymorphicWrapper<PyDynamics<Base>, Base> {
  using Scalar = typename Base::Scalar;
  using Data = DynamicsDataTpl<Scalar>;
  using Base::Base;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, y, boost::ref(data));
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, y,
                                  boost::ref(data));
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
                                    const ConstVectorRef &lbda,
                                    Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, Base, computeVectorHessianProducts, x, u, y,
                             lbda, boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, Base, createData, );
  }

  shared_ptr<Data> default_createData() const { return Base::createData(); }
};

} // namespace python
} // namespace aligator

namespace boost::python::objects {

template <>
struct value_holder<aligator::python::PyDynamics<>>
    : proxsuite::nlp::python::OwningNonOwningHolder<
          aligator::python::PyDynamics<>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

template <>
struct value_holder<
    aligator::python::PyDynamics<aligator::context::IntegratorAbstract>>
    : proxsuite::nlp::python::OwningNonOwningHolder<
          aligator::python::PyDynamics<aligator::context::IntegratorAbstract>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

} // namespace boost::python::objects
