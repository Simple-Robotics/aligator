#pragma once

#include "proxddp/modelling/dynamics/integrator-base.hpp"


namespace proxddp
{
  namespace dynamics
  {
    template<typename _Scalar>
    struct IntegratorRK2 : ExplicitIntegratorTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
      using Base = ExplicitIntegratorTpl<Scalar>;
      using Data = typename Base::Data;
    };
    
  } // namespace dynamics
} // namespace proxddp

