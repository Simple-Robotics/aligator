#pragma once

#include "proxddp/modelling/linear-function.hpp"

namespace proxddp
{
  /** @brief A simple function \f$f(u) = [u_{\min} - u; u - u_{\max}]\f$.
   */
  template<typename Scalar>
  struct ControlBoxFunction : LinearFunction<Scalar>
  {
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    VectorXs umin_, umax_;

    ControlBoxFunction(const int ndx, const VectorXs umin, const VectorXs umax)
      : LinearFunction<Scalar>(ndx, umin.size(), ndx, 2 * umin.size())
      , umin_(umin)
      , umax_(umax)
    {
      assert(umin.size() == umax.size() && "Size of umin and umax should be the same!");
      this->d_ << umin_, -umax_;
      this->B_.topRows(this->nu).diagonal().array() = -1.;
      this->B_.bottomRows(this->nu).setIdentity();
    }

    ControlBoxFunction(const int ndx, const int nu, Scalar umin, Scalar umax)
      : ControlBoxFunction(ndx, VectorXs::Constant(nu, umin), VectorXs::Constant(nu, umax))
      {}
  };
  
} // namespace proxddp

