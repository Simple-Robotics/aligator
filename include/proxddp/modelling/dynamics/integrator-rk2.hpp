#pragma once

#include "proxddp/modelling/dynamics/integrator-explicit.hpp"


namespace proxddp
{
  namespace dynamics
  {
    /** @brief  Second-order Runge-Kutta integrator.
     * 
     * \f{eqnarray*}{
     *    x_{k+1} = x_k \oplus h f(x^{(1)}, u_k),\\
     *    x^{(1)} = x_k \oplus \frac h2 f(x_k, u_k)
     * \f}
     * 
     */
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

