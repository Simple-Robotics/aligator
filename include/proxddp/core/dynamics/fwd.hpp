#pragma once

#include "proxddp/core/dynamics.hpp"


namespace proxddp
{

  /// @brief  Namespace for modelling system dynamics.
  namespace dynamics
  {

    // fwd ContinuousDynamicsTpl
    template<typename _Scalar>
    struct ContinuousDynamicsTpl;

    // fwd ContinuousDynamicsDataTpl
    template<typename _Scalar>
    struct ContinuousDynamicsDataTpl;

    // fwd ODEBaseTpl
    template<typename _Scalar>
    struct ODEBaseTpl;

    template<typename _Scalar>
    struct ODEDataTpl;

    //// INTEGRATORS

    // fwd IntegratorBaseTpl;
    template<typename _Scalar>
    struct IntegratorBaseTpl;

    // fwd IntegratorBaseDataTpl;
    template<typename _Scalar>
    struct IntegratorBaseDataTpl;

    // fwd ExplicitIntegratorTpl;
    template<typename _Scalar>
    struct ExplicitIntegratorTpl;

    // fwd ExplicitIntegratorDataTpl;
    template<typename _Scalar>
    struct ExplicitIntegratorDataTpl;

  } // namespace dynamics
  
} // namespace proxddp

