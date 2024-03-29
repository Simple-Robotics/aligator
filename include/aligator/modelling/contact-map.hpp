#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

/// @brief Contact map for centroidal costs and dynamics.
template <typename _Scalar> struct ContactMapTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using PoseVec = StdVectorEigenAligned<Vector3s>;

  ContactMapTpl(const std::vector<bool> &contact_states,
                const PoseVec &contact_poses)
      : contact_states_(contact_states), contact_poses_(contact_poses) {
    if (contact_states.size() != contact_poses.size()) {
      ALIGATOR_DOMAIN_ERROR(
          fmt::format("contact_states and contact_poses should have same size, "
                      "currently ({} and {}).",
                      contact_states.size(), contact_poses.size()));
    }
    size_ = contact_states_.size();
  }

  void addContact(const bool state, const Vector3s &pose) {
    contact_states_.push_back(state);
    contact_poses_.push_back(pose);
    size_ += 1;
  }

  void removeContact(const size_t i) {
    if (size_ == 0) {
      ALIGATOR_RUNTIME_ERROR("ContactMap is empty!");
    } else {
      contact_states_.erase(contact_states_.begin() + long(i));
      contact_poses_.erase(contact_poses_.begin() + long(i));
      size_ -= 1;
    }
  }

  const std::vector<bool> &getContactStates() const { return contact_states_; }

  bool getContactState(const std::size_t i) const { return contact_states_[i]; }

  const PoseVec &getContactPoses() const { return contact_poses_; }

  const Vector3s &getContactPose(const std::size_t i) const {
    return contact_poses_[i];
  }

  std::size_t getSize() const { return size_; }

private:
  std::vector<bool> contact_states_;
  PoseVec contact_poses_;
  std::size_t size_;
};

} // namespace aligator
