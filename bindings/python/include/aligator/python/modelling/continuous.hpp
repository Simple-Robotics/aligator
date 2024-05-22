/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator {
namespace python {

template <class T = dynamics::ContinuousDynamicsAbstractTpl<context::Scalar>>
struct PyContinuousDynamics : T, bp::wrapper<T> {
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(context::Scalar);

  template <class... Args> PyContinuousDynamics(Args &&...args) : T(args...) {}

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

template <class T = dynamics::ODEAbstractTpl<context::Scalar>>
struct PyODEAbstract : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;

  template <class... Args> PyODEAbstract(Args &&...args) : T(args...) {}

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
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
