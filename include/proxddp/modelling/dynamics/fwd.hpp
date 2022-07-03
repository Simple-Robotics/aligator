#pragma once

#include "proxddp/core/dynamics.hpp"

namespace proxddp {

/// @brief  Namespace for modelling system dynamics.
namespace dynamics {

// fwd ContinuousDynamicsAbstractTpl
template <typename _Scalar> struct ContinuousDynamicsAbstractTpl;

// fwd ContinuousDynamicsDataTpl
template <typename _Scalar> struct ContinuousDynamicsDataTpl;

// fwd ODEAbstractTpl
template <typename _Scalar> struct ODEAbstractTpl;

template <typename _Scalar> struct ODEDataTpl;

//// INTEGRATORS

// fwd IntegratorAbstractTpl;
template <typename _Scalar> struct IntegratorAbstractTpl;

// fwd IntegratorDataTpl;
template <typename _Scalar> struct IntegratorDataTpl;

// fwd ExplicitIntegratorAbstractTpl;
template <typename _Scalar> struct ExplicitIntegratorAbstractTpl;

// fwd ExplicitIntegratorDataTpl;
template <typename _Scalar> struct ExplicitIntegratorDataTpl;

// fwd IntegratorEulerTpl;
template <typename _Scalar> struct IntegratorEulerTpl;

// fwd IntegratorSemiImplEulerTpl;
template <typename _Scalar> struct IntegratorSemiImplEulerTpl;

// fwd IntegratorRK2Tpl;
template <typename _Scalar> struct IntegratorRK2Tpl;

} // namespace dynamics

} // namespace proxddp
