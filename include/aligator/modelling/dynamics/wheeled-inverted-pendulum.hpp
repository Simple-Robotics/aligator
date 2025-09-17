#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator::dynamics {

template <typename _Scalar>
struct WheeledInvertedPendulumDynamicsTpl : ODEAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  using Base = dynamics::ODEAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ODEData = dynamics::ContinuousDynamicsDataTpl<Scalar>;
  using VectorSpace = aligator::VectorSpaceTpl<Scalar, 7>;
  WheeledInvertedPendulumDynamicsTpl(const double gravity, const double length)
      : Base(VectorSpace{}, 2)
      , length_(length)
      , gravity_(gravity) {}

  double length_;
  double gravity_;

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ODEData &data) const override {
    Scalar rdot = x[0], phidot = x[1], theta = x[2], thetadot = x[3],
           phi = x[4];
    Scalar rdotdot = u[0], phidotdot = u[1];

    data.xdot_[0] = rdotdot;
    data.xdot_[1] = phidotdot;
    data.xdot_[2] = thetadot;
    data.xdot_[3] = std::sin(theta) * gravity_ / length_ -
                    std::cos(theta) * rdotdot / length_;
    data.xdot_[4] = phidot;
    data.xdot_[5] = rdot * std::cos(phi);
    data.xdot_[6] = rdot * std::sin(phi);
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ODEData &data) const override {
    Scalar rdot = x[0], phidot = x[1], theta = x[2], thetadot = x[3],
           phi = x[4], rdotdot = u[0];

    data.Jx_.setZero();
    data.Jx_(2, 3) = 1;
    data.Jx_(3, 2) = std::cos(theta) * gravity_ / length_ +
                     std::sin(theta) * rdotdot / length_;

    data.Jx_(4, 1) = 1;
    data.Jx_(5, 0) = std::cos(phi);
    data.Jx_(5, 4) = -rdot * std::sin(phi);
    data.Jx_(6, 0) = std::sin(phi);
    data.Jx_(6, 4) = rdot * std::cos(phi);

    data.Ju_.setZero();
    data.Ju_(0, 0) = 1;
    data.Ju_(1, 1) = 1;
    data.Ju_(3, 0) = -1 * std::cos(theta) / length_;
  }
};

} // namespace aligator::dynamics
