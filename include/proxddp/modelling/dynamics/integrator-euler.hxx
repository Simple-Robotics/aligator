#pragma once

#include "proxddp/modelling/dynamics/integrator-euler.hpp"


namespace proxddp
{
  namespace dynamics
  {
    template<typename Scalar>
    IntegratorEuler<Scalar>::IntegratorEuler(const shared_ptr<ODEType>& cont_dynamics, const Scalar timestep)
      : Base(cont_dynamics), timestep_(timestep) {}

    template<typename Scalar>
    void IntegratorEuler<Scalar>::
    forward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const
    {
      Data& d = static_cast<Data&>(data);
      ODEDataTpl<Scalar>& cdata = static_cast<ODEDataTpl<Scalar>&>(*d.continuous_data);
      this->ode_->forward(x, u, cdata);
      // dx = (dx/dt) * dt
      d.dx_ = cdata.xdot_ * timestep_;
      this->out_space().integrate(x, d.dx_, d.xout_);
    }

    
    template<typename Scalar>
    void IntegratorEuler<Scalar>::
    dForward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const
    {
      Data& d = static_cast<Data&>(data);
      ODEDataTpl<Scalar>& cdata = static_cast<ODEDataTpl<Scalar>&>(*d.continuous_data);
      this->ode_->dForward(x, u, cdata);
      d.Jx_ = timestep_ * cdata.Jx_; // dddx_dx
      d.Ju_ = timestep_ * cdata.Ju_; // ddx_du
      this->out_space().JintegrateTransport(x, d.dx_, d.Jx_, 1);
      this->out_space().JintegrateTransport(x, d.dx_, d.Ju_, 1);
      this->out_space().Jintegrate(x, d.dx_, d.Jtemp_, 0);
      d.Jx_ = d.Jtemp_ + d.Jx_; 
    }
  } // namespace dynamics
  
  
} // namespace proxddp

