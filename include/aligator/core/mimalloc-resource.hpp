/// @copyright Copyright (C) 2025 INRIA
#pragma once

#include <memory_resource>

namespace aligator {

/// @brief A memory_resource wrapping around mimalloc.
class mimalloc_resource : public std::pmr::memory_resource {
public:
  mimalloc_resource() = default;

private:
  [[nodiscard]] void *do_allocate(size_t bytes, size_t alignment) override;

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override;

  bool do_is_equal(const memory_resource &other) const noexcept override {
    // Check if 'other' is also a mimalloc_resource
    return dynamic_cast<const mimalloc_resource *>(&other) != nullptr;
  }
};

} // namespace aligator
