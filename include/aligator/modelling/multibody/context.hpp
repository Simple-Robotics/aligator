#pragma once

#include "aligator/context.hpp"

#include <pinocchio/fwd.hpp>

namespace aligator {
template <typename> struct MultibodyConfiguration;
template <typename> struct MultibodyPhaseSpace;

namespace context {
using PinModel = pinocchio::ModelTpl<Scalar, Options>;
using PinData = pinocchio::DataTpl<Scalar, Options>;

using RCM = pinocchio::RigidConstraintModelTpl<Scalar, Options>;
using RCD = pinocchio::RigidConstraintDataTpl<Scalar, Options>;
using MultibodyConfiguration = MultibodyConfiguration<Scalar>;
using MultibodyPhaseSpace = MultibodyPhaseSpace<Scalar>;
} // namespace context
} // namespace aligator
