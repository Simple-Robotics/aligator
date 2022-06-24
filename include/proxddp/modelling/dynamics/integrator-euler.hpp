#pragma once
/// @file integrator-euler.hpp
/// @brief Define the explicit Euler integrator.

#include "proxddp/modelling/dynamics/integrator-explicit.hpp"


namespace proxddp
{
  namespace dynamics
  {
    /// @brief Explicit Euler integrator \f$ x_{k+1} = x_k \oplus h f(x_k, u_k)\f$.
    template<typename _Scalar>
    struct IntegratorEuler : ExplicitIntegratorAbstractTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
      using Base = ExplicitIntegratorAbstractTpl<Scalar>;
      using Data = ExplicitIntegratorDataTpl<Scalar>;
      using ODEType = ODEAbstractTpl<Scalar>;

      using Base::evaluate;
      using Base::computeJacobians;
      using Base::computeVectorHessianProducts;

      /// Integration time step \f$h\f$.
      Scalar timestep_;

      explicit IntegratorEuler(const ODEType& cont_dynamics, const Scalar timestep)
        : Base(cont_dynamics), timestep_(timestep) {}

      void forward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const
      {
        Data& d = static_cast<Data&>(data);
        ODEDataTpl<Scalar>& cdata = static_cast<ODEDataTpl<Scalar>&>(*d.continuous_data);
        this->cont_dynamics_->forward(x, u, cdata);
        // dx = (dx/dt) * dt
        d.dx_ = cdata.xdot_ * timestep_;
        this->out_space().integrate(x, d.dx_, d.xout_);
      }

      void dForward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const
      {
        Data& d = static_cast<Data&>(data);
        ODEDataTpl<Scalar>& cdata = static_cast<ODEDataTpl<Scalar>&>(*d.continuous_data);
        this->cont_dynamics_->dForward(x, u, *cdata);
        d.Jx_ = timestep_ * cdata->Jx_; // dddx_dx
        d.Ju_ = timestep_ * cdata->Ju_; // ddx_du
        this->out_space().JintegrateTransport(x, d.dx_, d.Jx_, 1);
        this->out_space().JintegrateTransport(x, d.dx_, d.Ju_, 1);
        this->out_space().Jintegrate(x, d.dx_, d.Jtemp_, 0);
        d.Jx_ = d.Jtemp_ + d.Jx_; 
      }

    };
    
  } // namespace dynamics
} // namespace proxddp

