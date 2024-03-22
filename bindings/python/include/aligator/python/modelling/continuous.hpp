/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/continuous-base.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator {
namespace python {
namespace internal {

template <class T = dynamics::ContinuousDynamicsAbstractTpl<context::Scalar>>
struct PyContinuousDynamics : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  using Data = dynamics::ContinuousDynamicsDataTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  template <class... Args> PyContinuousDynamics(Args &&...args) : T(args...) {}

  virtual void
  configure(CommonModelBuilderContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, T, configure, container);
  }

  void default_configure(CommonModelBuilderContainer &container) const {
    T::configure(container);
  }

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

  virtual shared_ptr<Data>
  createData(const CommonModelDataContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE_DIFF_NAME(shared_ptr<Data>, T, createData,
                                       createDataWithCommon, container);
  }

  shared_ptr<Data> default_createDataWithCommon(
      const CommonModelDataContainer &container) const {
    return T::createData(container);
  }
};

template <class T = dynamics::ODEAbstractTpl<context::Scalar>>
struct PyODEAbstract : T, bp::wrapper<T> {
  using Scalar = context::Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ODEData = dynamics::ODEDataTpl<context::Scalar>;
  using Data = dynamics::ContinuousDynamicsDataTpl<context::Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  template <class... Args> PyODEAbstract(Args &&...args) : T(args...) {}

  virtual void
  configure(CommonModelBuilderContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, T, configure, container);
  }

  void default_configure(CommonModelBuilderContainer &container) const {
    return T::configure(container);
  }

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       ODEData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        ODEData &data) const override {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, boost::ref(data));
  }

  shared_ptr<Data> createData() const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, );
  }

  shared_ptr<Data> default_createData() const { return T::createData(); }

  virtual shared_ptr<Data>
  createData(const CommonModelDataContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, T, createData, container);
  }

  shared_ptr<Data> default_createDataWithCommon(
      const CommonModelDataContainer &container) const {
    return T::createData(container);
  }
};

} // namespace internal

} // namespace python
} // namespace aligator
