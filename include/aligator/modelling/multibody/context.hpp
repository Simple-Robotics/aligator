#pragma once

#include "aligator/context.hpp"

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

namespace aligator {
namespace context {
using PinModel = pinocchio::ModelTpl<Scalar, Options>;
using PinData = pinocchio::DataTpl<Scalar, Options>;

using RCM = pinocchio::RigidConstraintModelTpl<Scalar, Options>;
using RCD = pinocchio::RigidConstraintDataTpl<Scalar, Options>;
using MultibodyConfiguration = proxsuite::nlp::MultibodyConfiguration<Scalar>;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<Scalar>;
} // namespace context
} // namespace aligator
