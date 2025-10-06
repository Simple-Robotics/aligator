/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/explicit-dynamics.hpp"

#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/integrator-explicit.hpp"

namespace aligator {
namespace python {
using context::DynamicsData;

/// Wrapper for ExplicitDynamicsModel which avoids redeclaring overrides for any
/// child virtual class (e.g. integrator classes).
/// @tparam ExplicitBase The derived virtual class that is being exposed.
/// @sa PyStageFunction
template <class ExplicitBase = context::ExplicitDynamics>
struct PyExplicitDynamics final
    : ExplicitBase,
      PolymorphicWrapper<PyExplicitDynamics<ExplicitBase>, ExplicitBase> {
  using Scalar = context::Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  // All functions in the interface take this type for output
  using Data = ExplicitDynamicsDataTpl<Scalar>;

  using ExplicitBase::ExplicitBase;

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               Data &data) const {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "forward", x, u, boost::ref(data));
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                Data &data) const {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, boost::ref(data));
  }

  shared_ptr<Data> createData() const {
    ALIGATOR_PYTHON_OVERRIDE(shared_ptr<Data>, ExplicitBase, createData, );
  }

  shared_ptr<Data> default_createData() const {
    return ExplicitBase::createData();
  }
};

} // namespace python
} // namespace aligator

namespace boost::python::objects {

template <>
struct value_holder<aligator::python::PyExplicitDynamics<>>
    : aligator::python::OwningNonOwningHolder<
          aligator::python::PyExplicitDynamics<>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

template <>
struct value_holder<aligator::python::PyExplicitDynamics<
    aligator::context::ExplicitIntegratorAbstract>>
    : aligator::python::OwningNonOwningHolder<
          aligator::python::PyExplicitDynamics<
              aligator::context::ExplicitIntegratorAbstract>> {
  using OwningNonOwningHolder::OwningNonOwningHolder;
};

} // namespace boost::python::objects
