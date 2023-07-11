#pragma once

#include "proxddp/context.hpp"

#include <proxnlp/modelling/spaces/multibody.hpp>

namespace proxddp {
namespace context {
using PinModel = pinocchio::ModelTpl<Scalar, Options>;
using PinData = pinocchio::DataTpl<Scalar, Options>;
#ifdef PROXDDP_PINOCCHIO_V3
using RCM = pinocchio::RigidConstraintModelTpl<Scalar, Options>;
using RCD = pinocchio::RigidConstraintDataTpl<Scalar, Options>;
#endif
using MultibodyConfiguration = proxnlp::MultibodyConfiguration<Scalar>;
using MultibodyPhaseSpace = proxnlp::MultibodyPhaseSpace<Scalar>;
} // namespace context
} // namespace proxddp
