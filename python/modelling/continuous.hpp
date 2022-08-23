#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/modelling/dynamics/continuous-base.hpp"
#include "proxddp/modelling/dynamics/ode-abstract.hpp"

namespace proxddp {
namespace python {
namespace internal {

template <class T = dynamics::ContinuousDynamicsAbstractTpl<context::Scalar>>
struct PyContinuousDynamics : T, bp::wrapper<T> {
  using bp::wrapper<T>::get_override;
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;
  PROXNLP_DYNAMIC_TYPEDEFS(context::Scalar);

  using T::T;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &xdot, Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, xdot, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xdot, Data &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, xdot, data);
  }

  shared_ptr<Data> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

template <class T = dynamics::ODEAbstractTpl<context::Scalar>>
struct PyODEAbstract : T, bp::wrapper<T> {
  using bp::wrapper<T>::get_override;
  using Scalar = context::Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ODEData = dynamics::ODEDataTpl<context::Scalar>;
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;

  using T::T;

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       ODEData &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "forward", x, u, data);
  }

  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        ODEData &data) const override {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, data);
  }

  shared_ptr<Data> createData() const override {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }
};

} // namespace internal

} // namespace python
} // namespace proxddp
