#pragma once

#include <math.h>
#include "aligator/core/vector-space.hpp"

using T = double;
using VEC = aligator::VectorSpaceTpl<T>;

using namespace aligator;
ALIGATOR_DYNAMIC_TYPEDEFS(T);

template <typename T>
struct WheeledInvertedPendulumDynamicsTpl : dynamics::ODEAbstractTpl<T> {
  using Base = dynamics::ODEAbstractTpl<T>;
  using ODEData = dynamics::ContinuousDynamicsDataTpl<T>;
  WheeledInvertedPendulumDynamicsTpl(const double gravity, const double length)
      : Base(VEC(4), 2)
      , length_(length)
      , gravity_(gravity) {}

  double length_;
  double gravity_;

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ODEData &data) const override {
    T rdot = x[0], phidot = x[1], theta = x[2], thetadot = x[3];
    T rdotdot = u[0], phidotdot = u[1];

    data.xdot_[0] = rdotdot;
    data.xdot_[1] = phidotdot;
    data.xdot_[2] = thetadot;
    data.xdot_[3] = std::sin(theta) * gravity_ / length_ -
                    std::cos(theta) * rdotdot / length_;
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ODEData &data) const override {
    T theta = x[2], rdotdot = u[0];

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
