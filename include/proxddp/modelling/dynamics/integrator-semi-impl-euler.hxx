#pragma once

#include "proxddp/modelling/dynamics/integrator-semi-impl-euler.hpp"


namespace proxddp
{
  namespace dynamics
  {
    template<typename Scalar>
    IntegratorSemiImplEulerTpl<Scalar>::IntegratorSemiImplEulerTpl(const shared_ptr<ODEType>& cont_dynamics, const Scalar timestep)
      : Base(cont_dynamics), timestep_(timestep) {}

    template<typename Scalar>
    void IntegratorSemiImplEulerTpl<Scalar>::
    forward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const
    {
      Data& d = static_cast<Data&>(data);
      ODEDataTpl<Scalar>& cdata = static_cast<ODEDataTpl<Scalar>&>(*d.continuous_data);
      this->ode_->forward(x, u, cdata);
      // dx = (dx/dt) * dt
      int ndx =  this->ndx1;
      for(int i=0; i<ndx/2; i++){
        d.dx_(ndx/2+i) = cdata.xdot_(i) * timestep_;
      }
      this->out_space().integrate(x, d.dx_, d.xnext_);
      for(int i=0; i<ndx/2; i++){
        d.dx_(i) = d.xnext_(ndx/2+i);
      }
      this->out_space().integrate(x, d.dx_, d.xnext_);
    }

    
    template<typename Scalar>
    void IntegratorSemiImplEulerTpl<Scalar>::
    dForward(const ConstVectorRef& x, const ConstVectorRef& u, ExplicitDynamicsDataTpl<Scalar>& data) const
    {
      Data& d = static_cast<Data&>(data);
      ODEDataTpl<Scalar>& cdata = static_cast<ODEDataTpl<Scalar>&>(*d.continuous_data);
      int ndx =  this->ndx1;
      int nu =  this->nu;
      MatrixXs Jxtemp_(ndx, ndx), Jutemp_(ndx,nu);
      this->ode_->dForward(x, u, cdata);
      // dv_dx and dv_du are same as euler explicit
      d.Jx_ = timestep_ * cdata.Jx_; // dddx_dx
      d.Ju_ = timestep_ * cdata.Ju_; // ddx_du
      this->out_space().JintegrateTransport(x, d.dx_, d.Jx_, 1);
      this->out_space().JintegrateTransport(x, d.dx_, d.Ju_, 1);
      this->out_space().Jintegrate(x, d.dx_, d.Jtmp_xnext, 0);
      d.Jx_ = d.Jtmp_xnext + d.Jx_; 
      // dq_dx and dq_du needs to be modified
      for(int i = 0; i<ndx/2; i++){
        Jxtemp_(i) = timestep_ *d.Jx_(ndx/2+i);
        Jxtemp_(i+ndx/2) = timestep_ *cdata.Jx_(ndx/2+i);
      }
      for(int i = 0; i<ndx/2; i++){
        Jutemp_(i) = timestep_ *d.Ju_(ndx/2+i);
        Jutemp_(i+ndx/2) = timestep_ *cdata.Ju_(ndx/2+i);
      }
      this->out_space().JintegrateTransport(x, d.dx_, Jxtemp_, 1);
      this->out_space().JintegrateTransport(x, d.dx_, Jutemp_, 1);
      Jxtemp_ = d.Jtmp_xnext + Jxtemp_; 
      for(int i = 0; i<ndx/2; i++){
        d.Jx_(i) = Jxtemp_(i);
      }
      for(int i = 0; i<ndx/2; i++){
        d.Ju_(i) = Jutemp_(i);
      }
    }
  } // namespace dynamics
  
  
} // namespace proxddp
