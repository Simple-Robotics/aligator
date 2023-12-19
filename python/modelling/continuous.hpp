/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/modelling/dynamics/continuous-base.hpp"
#include "proxddp/modelling/dynamics/ode-abstract.hpp"

namespace proxddp {
namespace python {
namespace internal {

template <class T = dynamics::ContinuousDynamicsAbstractTpl<context::Scalar>>
struct PyContinuousDynamics : T, bp::wrapper<T> {
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;
  PROXDDP_DYNAMIC_TYPEDEFS(context::Scalar);

  template <class... Args> PyContinuousDynamics(Args &&...args) : T(args...) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &xdot, Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, xdot,
                                 boost::ref(data));
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xdot, Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, xdot,
                                 boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

template <class T = dynamics::ODEAbstractTpl<context::Scalar>>
struct PyODEAbstract : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using ODEData = dynamics::ODEDataTpl<context::Scalar>;
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;

  template <class... Args> PyODEAbstract(Args &&...args) : T(args...) {}

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       ODEData &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        ODEData &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

} // namespace internal

} // namespace python
} // namespace proxddp
