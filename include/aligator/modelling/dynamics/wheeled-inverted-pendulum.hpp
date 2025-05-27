#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator::dynamics {

template <typename _Scalar>
struct WheeledInvertedPendulumDynamicsTpl : ODEAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  using Base = dynamics::ODEAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ODEData = dynamics::ContinuousDynamicsDataTpl<Scalar>;
  using VectorSpace = aligator::VectorSpaceTpl<Scalar, 4>;
  WheeledInvertedPendulumDynamicsTpl(const double gravity, const double length)
      : Base(VectorSpace{}, 2)
      , length_(length)
      , gravity_(gravity) {}

  double length_;
  double gravity_;

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ODEData &data) const override {
    Scalar rdot = x[0], phidot = x[1], theta = x[2], thetadot = x[3];
    Scalar rdotdot = u[0], phidotdot = u[1];

    data.xdot_[0] = rdotdot;
    data.xdot_[1] = phidotdot;
    data.xdot_[2] = thetadot;
    data.xdot_[3] = std::sin(theta) * gravity_ / length_ -
                    std::cos(theta) * rdotdot / length_;
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ODEData &data) const override {
    Scalar theta = x[2], rdotdot = u[0];

    data.Jx_.setZero();
    data.Jx_(3, 2) = 1;
    data.Jx_(2, 3) = std::cos(theta) * gravity_ / length_ +
                     std::sin(theta) * rdotdot / length_;

    data.Ju_.setZero();
    data.Ju_(0, 0) = 1;
    data.Ju_(1, 1) = 1;
    data.Ju_(3, 0) = -1 * std::cos(theta) / length_;
  }
};

} // namespace aligator::dynamics
