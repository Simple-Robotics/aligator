#pragma once

#include "proxddp/modelling/dynamics/fwd.hpp"


namespace proxddp
{
  namespace dynamics
  {
    /// @brief Base class for ODE dynamics \f$ \dot{x} = f(x, u) \f$.
    template<typename _Scalar>
    struct ODEBaseTpl : ContinuousDynamicsTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

      using BaseType = ContinuousDynamicsTpl<_Scalar>;
      using BaseType::BaseType;

    };
    
  } // namespace dynamics
} // namespace proxddp

