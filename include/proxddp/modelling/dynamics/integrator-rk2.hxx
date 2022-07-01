#pragma once

#include "proxddp/modelling/dynamics/integrator-rk2.hpp"

namespace proxddp
{
  namespace dynamics
  {
    template<typename Scalar>
    IntegratorRK2Tpl<Scalar>::
    IntegratorRK2Tpl(const shared_ptr<ODEType>& cont_dynamics, const Scalar timestep)
      : Base(cont_dynamics)
      , timestep_(timestep)
      {}
    
    template<typename Scalar>
    void IntegratorRK2Tpl<Scalar>::
    forward(const ConstVectorRef& x, const ConstVectorRef& u, BaseData& data) const
    {
      Data& d = static_cast<Data&>(data);
      using ODEData = ODEDataTpl<Scalar>;
      ODEData& cd1 = static_cast<ODEData&>(*d.continuous_data);
      ODEData& cd2 = static_cast<ODEData&>(*d.continuous_data2);

      this->ode_->forward(x, u, cd1);
      d.dx1_ = dt_2_ * cd1.xdot_;
      this->next_state_->integrate(x, d.dx1_, d.x1_);

      this->ode_->forward(d.x1_, u, cd2);
      d.dx_ = timestep_ * cd2.xdot_;
      this->next_state_->integrate(x, d.dx_, d.xnext_);
    }

    template<typename Scalar>
    void IntegratorRK2Tpl<Scalar>::
    dForward(const ConstVectorRef& x, const ConstVectorRef& u, BaseData& data) const
    {
      Data& d = static_cast<Data&>(data);
      using ODEData = ODEDataTpl<Scalar>;
      ODEData& cd1 = static_cast<ODEData&>(*d.continuous_data);
      ODEData& cd2 = static_cast<ODEData&>(*d.continuous_data2);

      // dx1_dz = Transport(d(dx1)_dz)
      this->ode_->dForward(x, u, cd1);
      d.Jx_ = dt_2_ * cd1.Jx_;
      d.Ju_ = dt_2_ * cd1.Ju_;
      this->out_space().JintegrateTransport(x, d.dx1_, d.Jx_, 1);
      this->out_space().JintegrateTransport(x, d.dx1_, d.Ju_, 1);

      // J = d(dx)_dz = d(dx)_dx1 * dx1_dz
      // then transport J to xnext = exp(dx) * x1
      this->ode_->dForward(d.x1_, u, cd2);
      d.Jx_ = (timestep_ * cd2.Jx_) * d.Jx_;
      d.Ju_ = (timestep_ * cd2.Ju_) * d.Ju_;
      this->out_space().JintegrateTransport(d.x1_, d.dx_, d.Jx_, 1);
      this->out_space().JintegrateTransport(d.x1_, d.dx_, d.Ju_, 1);

      this->out_space().Jintegrate(d.x1_, d.dx_, d.Jtmp_xnext, 0);
      d.Jx_.noalias() += d.Jtmp_xnext;
    }

    template<typename Scalar>
    RK2DataTpl<Scalar>::RK2DataTpl(const IntegratorRK2Tpl<Scalar>* integrator)
      : Base(integrator)
      , x1_(integrator->out_space().neutral())
      {
        continuous_data2 = std::make_shared<ODEData>(*this->continuous_data);
      }


  } // namespace dynamics
} // namespace proxddp
