#pragma once

#include "proxddp/context.hpp"
#include <pinocchio/fwd.hpp>

namespace proxddp {
namespace context {
using PinModel = pinocchio::ModelTpl<Scalar, Options>;
using PinData = pinocchio::DataTpl<Scalar, Options>;
using RCM = pinocchio::RigidConstraintModelTpl<Scalar, Options>;
using RCD = pinocchio::RigidConstraintDataTpl<Scalar, Options>;
} // namespace context
} // namespace proxddp
