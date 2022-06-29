#pragma once
/// @file integrator-euler.hpp
/// @brief Define the explicit Euler integrator.

#include "proxddp/modelling/dynamics/integrator-explicit.hpp"


namespace proxddp
{
  namespace dynamics
  {
    /**
     *  @brief Explicit Euler integrator \f$ x_{k+1} = x_k \oplus h f(x_k, u_k)\f$.
     */
    template<typename _Scalar>
    struct IntegratorEuler : ExplicitIntegratorAbstractTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
      using Base = ExplicitIntegratorAbstractTpl<Scalar>;
      using Data = ExplicitIntegratorDataTpl<Scalar>;
      using ODEType = ODEAbstractTpl<Scalar>;

      /// Integration time step \f$h\f$.
      Scalar timestep_;

      explicit IntegratorEuler(const shared_ptr<ODEType>& cont_dynamics, const Scalar timestep);

      void forward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const;

      void dForward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const;

    };
    
  } // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/integrator-euler.hxx"
