#pragma once

#include <pinocchio/algorithm/frames.hpp>

namespace aligator {

struct frame_api {
  pinocchio::FrameIndex getFrameId() const { return pin_frame_id_; }
  void setFrameId(const std::size_t id) { pin_frame_id_ = id; }

protected:
  pinocchio::FrameIndex pin_frame_id_;
};

} // namespace aligator
