/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"

#include "proxsuite-nlp/python/polymorphic.hpp"

namespace aligator {
namespace python {

template <class T = context::ContinuousDynamicsAbstract>
struct PyContinuousDynamics final
    : T,
      proxsuite::nlp::python::PolymorphicWrapper<PyContinuousDynamics<T>, T> {
  using Data = context::ContinuousDynamicsData;
  ALIGATOR_DYNAMIC_TYPEDEFS(context::Scalar);
  using T::T;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &xdot, Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, xdot,
                                  boost::ref(data));
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xdot, Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, xdot,
                                  boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

template <class T = context::ODEAbstract>
struct PyODEAbstract final
    : T,
      proxsuite::nlp::python::PolymorphicWrapper<PyODEAbstract<T>, T> {
  using Scalar = context::Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = context::ODEData;

  using T::T;

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

} // namespace python
} // namespace aligator
//
namespace boost::python::objects {

template <>
struct value_holder<aligator::python::PyContinuousDynamics<>>
    : proxsuite::nlp::python::OwningNonOwningHolder<
          aligator::python::PyContinuousDynamics<>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

template <>
struct value_holder<aligator::python::PyODEAbstract<>>
    : proxsuite::nlp::python::OwningNonOwningHolder<
          aligator::python::PyODEAbstract<>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

} // namespace boost::python::objects
