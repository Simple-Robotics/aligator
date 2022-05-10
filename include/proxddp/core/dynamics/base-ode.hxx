/// @file base-ode.hxx  Implement the ContinuousDynamicsTpl interface for BaseODETpl.
#pragma once

#include "proxddp/core/dynamics/base-ode.hpp"

#include <proxnlp/manifold-base.hpp>


namespace proxddp
{
  namespace dynamics
  {

    template<typename Scalar>
    void ODEBaseTpl<Scalar>::evaluate(
      const ConstVectorRef& x,
      const ConstVectorRef& u,
      const ConstVectorRef& xdot,
      Data& data) const
    {
      auto d = static_cast<SpecificData&>(data);
      this->forward(x, u, d.xdot_);
      // xdot (-) computed xdot
      this->space_.difference(d.xdot_, xdot, d.error_);
    }

    template<typename Scalar>
    void ODEBaseTpl<Scalar>::computeJacobians(
      const ConstVectorRef& x,
      const ConstVectorRef& u,
      const ConstVectorRef&,
      Data& data) const
    {
      auto d = static_cast<SpecificData&>(data);
      this->dForward(x, u, d.Jx_, d.Ju_);
      d.Jxdot_.setIdentity();
      d.Jxdot_ *= static_cast<Scalar>(-1.);
    }

  } // namespace dynamics
} // namespace proxddp

