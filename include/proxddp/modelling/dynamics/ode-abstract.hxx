/// @file ode-abstract.hxx  Implement the ContinuousDynamicsAbstractTpl interface for BaseODETpl.
#pragma once

#include "proxddp/modelling/dynamics/ode-abstract.hpp"

#include <proxnlp/manifold-base.hpp>


namespace proxddp
{
  namespace dynamics
  {

    template<typename Scalar>
    void ODEAbstractTpl<Scalar>::evaluate(
      const ConstVectorRef& x,
      const ConstVectorRef& u,
      const ConstVectorRef& xdot,
      BaseData& data) const
    {
      Data& d = static_cast<Data&>(data);
      this->forward(x, u, d);
      d.value_ = d.xdot_ - xdot;
    }

    template<typename Scalar>
    void ODEAbstractTpl<Scalar>::computeJacobians(
      const ConstVectorRef& x,
      const ConstVectorRef& u,
      const ConstVectorRef&,
      BaseData& data) const
    {
      Data& d = static_cast<Data&>(data);
      this->dForward(x, u, d);
      d.Jxdot_.setIdentity();
      d.Jxdot_ *= -1.;
    }

  } // namespace dynamics
} // namespace proxddp

