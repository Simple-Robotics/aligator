#pragma once
///   Define the explicit Euler integrator.

#include "proxddp/modelling/dynamics/integrator-base.hpp"


namespace proxddp
{
  namespace dynamics
  {
    /// @brief Explicit Euler integrator \f$ x_{k+1} = x_k + h f(x_k, u_k)\f$.
    template<typename _Scalar>
    struct IntegratorEuler : ExplicitIntegratorTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
      using Base = ExplicitIntegratorTpl<Scalar>;
      using Data = ExplicitIntegratorDataTpl<Scalar>;

      /// Integration time step \f$h\f$.
      Scalar timestep_;

      IntegratorEuler(const typename Base::ContDynamics& cont_dynamics,
                          const Scalar timestep)
        : Base(cont_dynamics)
        , timestep_(timestep) {}

      void forward(const ConstVectorRef& x, const ConstVectorRef& u, VectorRef out) const
      {
        const auto& dyn = this->continuous();
        dyn.forward(x, u, out);
      }


    };
    
  } // namespace dynamics
} // namespace proxddp

