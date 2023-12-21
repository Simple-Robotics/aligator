#pragma once

#include "proxddp/core/dynamics.hpp"

namespace aligator {

/// @brief  Namespace for modelling system dynamics.
namespace dynamics {

// fwd ContinuousDynamicsAbstractTpl
template <typename Scalar> struct ContinuousDynamicsAbstractTpl;

// fwd ContinuousDynamicsDataTpl
template <typename Scalar> struct ContinuousDynamicsDataTpl;

// fwd ODEAbstractTpl
template <typename Scalar> struct ODEAbstractTpl;

template <typename Scalar> struct ODEDataTpl;

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

} // namespace aligator
