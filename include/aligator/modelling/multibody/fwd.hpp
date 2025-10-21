/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/context.hpp"
#include <pinocchio/algorithm/frames.hpp>

namespace aligator {

struct frame_api {
  pinocchio::FrameIndex getFrameId() const { return pin_frame_id_; }
  void setFrameId(const std::size_t id) { pin_frame_id_ = id; }

protected:
  pinocchio::FrameIndex pin_frame_id_;
};

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
