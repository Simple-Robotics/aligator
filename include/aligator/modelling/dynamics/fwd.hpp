/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2026 INRIA
#pragma once

#include "aligator/deprecated.hpp"
#include "aligator/context.hpp"

namespace aligator {

/// @brief  Namespace for modelling system dynamics.
namespace dynamics {

// fwd ContinuousDynamicsAbstractTpl
template <typename Scalar> struct ContinuousDynamicsAbstractTpl;

// fwd ContinuousDynamicsDataTpl
template <typename Scalar> struct ContinuousDynamicsDataTpl;

// fwd ODEAbstractTpl
template <typename Scalar> struct ODEAbstractTpl;

// fwd CentroidalFwdDynamicsTpl
template <typename Scalar> struct CentroidalFwdDynamicsTpl;

template <typename Scalar> struct CentroidalFwdDataTpl;

// fwd ContinuousCentroidalFwdDynamicsTpl
template <typename Scalar> struct ContinuousCentroidalFwdDynamicsTpl;

template <typename Scalar> struct ContinuousCentroidalFwdDataTpl;

//// INTEGRATORS

// fwd IntegratorAbstractTpl;
template <typename Scalar> struct IntegratorAbstractTpl;

// fwd IntegratorDataTpl;
template <typename Scalar> struct IntegratorDataTpl;

// fwd ExplicitIntegratorAbstractTpl;
template <typename Scalar> struct ExplicitIntegratorAbstractTpl;

// fwd ExplicitIntegratorDataTpl;
template <typename Scalar> struct ExplicitIntegratorDataTpl;

// fwd IntegratorEulerTpl;
template <typename Scalar> struct IntegratorEulerTpl;

// fwd IntegratorSemiImplEulerTpl;
template <typename Scalar> struct IntegratorSemiImplEulerTpl;

// fwd IntegratorSemiImplDataTpl;
template <typename Scalar> struct IntegratorSemiImplDataTpl;

// fwd IntegratorRK2Tpl;
template <typename Scalar> struct IntegratorRK2Tpl;

} // namespace dynamics

namespace context {
using ContinuousDynamicsAbstract =
    dynamics::ContinuousDynamicsAbstractTpl<Scalar>;
using ContinuousDynamicsData = dynamics::ContinuousDynamicsDataTpl<Scalar>;

using ODEAbstract = dynamics::ODEAbstractTpl<Scalar>;
using ODEData = dynamics::ContinuousDynamicsDataTpl<Scalar>;

using IntegratorAbstract = dynamics::IntegratorAbstractTpl<Scalar>;
using IntegratorData = dynamics::IntegratorDataTpl<Scalar>;

using ExplicitIntegratorAbstract =
    dynamics::ExplicitIntegratorAbstractTpl<Scalar>;
using ExplicitIntegratorData = dynamics::ExplicitIntegratorDataTpl<Scalar>;

} // namespace context
} // namespace aligator
