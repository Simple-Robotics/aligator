/// @copyright Copyright (C) 2024 INRIA
#pragma once

#include <Eigen/Core>
#include <memory_resource>

namespace aligator {

using byte_t = std::byte;

/// @brief A convenience subclass of @ref std::pmr::polymorphic_allocator for
/// bytes.
/// @details This subclass adds templated @ref allocate() and @ref deallocate()
/// methods which take the desired pointer alignment as argument. This extends
/// the standard C++ allocator API for convenience use with allocating buffers
/// in vectorized linear algbera.
class polymorphic_allocator : public std::pmr::polymorphic_allocator<byte_t> {
public:
  using base = std::pmr::polymorphic_allocator<byte_t>;
  polymorphic_allocator() noexcept : base{std::pmr::get_default_resource()} {}

  polymorphic_allocator(const polymorphic_allocator &other) = default;

  template <typename U>
  polymorphic_allocator(
      const std::pmr::polymorphic_allocator<U> &other) noexcept
      : base{other} {}

  polymorphic_allocator(std::pmr::memory_resource *resource) noexcept
      : std::pmr::polymorphic_allocator<byte_t>(resource) {}

  template <typename T>
  [[nodiscard]] T *allocate(size_t n,
                            size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return static_cast<T *>(resource()->allocate(n * sizeof(T), alignment));
  }

  template <typename T>
  void deallocate(T *p, size_t n,
                  size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, n * sizeof(T), alignment);
  }

  [[nodiscard]] void *
  allocate_bytes(size_t num_bytes,
                 size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return resource()->allocate(num_bytes, alignment);
  }

  void deallocate_bytes(void *p, size_t num_bytes,
                        size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, num_bytes, alignment);
  }
};

} // namespace aligator

template <>
struct std::allocator_traits<aligator::polymorphic_allocator>
    : std::allocator_traits<aligator::polymorphic_allocator::base> {
  using allocator_type = aligator::polymorphic_allocator;
};
