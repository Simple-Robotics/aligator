/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/explicit-dynamics.hpp"

namespace aligator {
namespace python {
namespace internal {

/// Wrapper for ExplicitDynamicsModel which avoids redeclaring overrides for any
/// child virtual class (e.g. integrator classes).
/// @tparam ExplicitBase The derived virtual class that is being exposed.
/// @sa PyStageFunction
template <class ExplicitBase = context::ExplicitDynamics>
struct PyExplicitDynamics : ExplicitBase, bp::wrapper<ExplicitBase> {
  using Scalar = context::Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  // All functions in the interface take this type for output
  using Data = ExplicitDynamicsDataTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  template <typename... Args>
  PyExplicitDynamics(Args &&...args) : ExplicitBase(args...) {}

  virtual void
  configure(CommonModelBuilderContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(void, ExplicitBase, configure, container);
  }

  void default_configure(CommonModelBuilderContainer &container) const {
    ExplicitBase::configure(container);
  }

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, boost::ref(data));
  }

  virtual shared_ptr<StageFunctionData>
  createData(const CommonModelDataContainer &container) const override {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<StageFunctionData>, ExplicitBase,
                             createData, container);
  }

  shared_ptr<StageFunctionData>
  default_createData(const CommonModelDataContainer &container) const {
    return ExplicitBase::createData(container);
  }
};

} // namespace internal
} // namespace python
} // namespace aligator
