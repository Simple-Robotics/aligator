#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

/// @brief Contact map for centroidal costs and dynamics.
template <typename _Scalar> struct ContactMapTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using PoseVec = StdVectorEigenAligned<Vector3s>;

  ContactMapTpl(const std::vector<std::string> &contact_names,
                const std::vector<bool> &contact_states,
                const PoseVec &contact_poses)
      : contact_names_(contact_names)
      , contact_states_(contact_states)
      , contact_poses_(contact_poses) {
    if (contact_states.size() != contact_poses.size()) {
      ALIGATOR_DOMAIN_ERROR(
          "Contact_states and contact_poses should have same size, "
          "currently ({:d} and {:d}).",
          contact_states.size(), contact_poses.size());
    }
    size_ = contact_states_.size();
  }

  void addContact(const std::string &name, const bool state,
                  const Vector3s &pose) {
    contact_names_.push_back(name);
    contact_states_.push_back(state);
    contact_poses_.push_back(pose);
    size_ += 1;
  }

  void removeContact(const size_t i) {
    if (size_ == 0) {
      ALIGATOR_RUNTIME_ERROR("ContactMap is empty!");
    } else {
      contact_names_.erase(contact_names_.begin() + long(i));
      contact_states_.erase(contact_states_.begin() + long(i));
      contact_poses_.erase(contact_poses_.begin() + long(i));
      size_ -= 1;
    }
  }

  bool getContactState(const std::string &name) const {
    auto id = std::find(contact_names_.begin(), contact_names_.end(), name);
    if (id == contact_names_.end()) {
      ALIGATOR_RUNTIME_ERROR("Contact name does not exist in this map!");
    }

    return contact_states_[id - contact_names_.begin()];
  }

  const Vector3s &getContactPose(const std::string &name) const {
    auto id = std::find(contact_names_.begin(), contact_names_.end(), name);
    if (id == contact_names_.end()) {
      ALIGATOR_RUNTIME_ERROR("Contact name does not exist in this map!");
    }

    return contact_poses_[id - contact_names_.begin()];
  }

  void setContactPose(const std::string &name, const Vector3s &ref) {
    auto id = std::find(contact_names_.begin(), contact_names_.end(), name);
    if (id == contact_names_.end()) {
      ALIGATOR_RUNTIME_ERROR("Contact name does not exist in this map!");
    }

    contact_poses_[id - contact_names_.begin()] = ref;
  }

  std::vector<std::string> contact_names_;
  std::vector<bool> contact_states_;
  PoseVec contact_poses_;
  std::size_t size_;
};

} // namespace aligator
