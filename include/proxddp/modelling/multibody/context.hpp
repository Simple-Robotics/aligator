#pragma once

#include "proxddp/context.hpp"

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

namespace proxddp {
namespace context {
using PinModel = pinocchio::ModelTpl<Scalar, Options>;
using PinData = pinocchio::DataTpl<Scalar, Options>;
#ifdef PROXDDP_PINOCCHIO_V3
using RCM = pinocchio::RigidConstraintModelTpl<Scalar, Options>;
using RCD = pinocchio::RigidConstraintDataTpl<Scalar, Options>;
#endif
using MultibodyConfiguration = proxsuite::nlp::MultibodyConfiguration<Scalar>;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<Scalar>;
} // namespace context
} // namespace proxddp
