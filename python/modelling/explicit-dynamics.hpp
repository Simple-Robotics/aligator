/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/python/functions.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

namespace proxddp {
namespace python {
namespace internal {

/// Wrapper for ExplicitDynamicsModel which avoids redeclaring overrides for any
/// child virtual class (e.g. integrator classes).
/// @tparam ExplicitBase The derived virtual class that is being exposed.
/// @sa PyStageFunction
template <class ExplicitBase = context::ExplicitDynamics>
struct PyExplicitDynamics : ExplicitBase, bp::wrapper<ExplicitBase> {
  using Scalar = context::Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  // All functions in the interface take this type for output
  using Data = ExplicitDynamicsDataTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;

  template <typename... Args>
  PyExplicitDynamics(Args &&...args) : ExplicitBase(args...) {}

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, boost::ref(data));
  }

  shared_ptr<FunctionData> createData() const {
    PROXDDP_PYTHON_OVERRIDE(shared_ptr<FunctionData>, ExplicitBase,
                            createData, );
  }

  shared_ptr<FunctionData> default_createData() const {
    return ExplicitBase::createData();
  }
};

} // namespace internal
} // namespace python
} // namespace proxddp
