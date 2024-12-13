#pragma once

#include <Eigen/Core>
#include <memory_resource>

namespace aligator {

using byte_t = unsigned char;
class polymorphic_allocator : public std::pmr::polymorphic_allocator<byte_t> {
public:
  polymorphic_allocator() noexcept
      : std::pmr::polymorphic_allocator<byte_t>(
            std::pmr::get_default_resource()) {}

  polymorphic_allocator(const polymorphic_allocator &other) = default;

  polymorphic_allocator(std::pmr::memory_resource *resource) noexcept
      : std::pmr::polymorphic_allocator<byte_t>(resource) {}

  template <typename T>
  T *allocate(size_t n, size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return static_cast<T *>(resource()->allocate(n * sizeof(T), alignment));
  }

  template <typename T>
  void deallocate(T *p, size_t n,
                  size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, n * sizeof(T), alignment);
  }

  void *allocate_bytes(size_t num_bytes,
                       size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    return resource()->allocate(num_bytes, alignment);
  }

  void deallocate_bytes(void *p, size_t num_bytes,
                        size_t alignment = EIGEN_DEFAULT_ALIGN_BYTES) {
    resource()->deallocate(p, num_bytes, alignment);
  }
};

} // namespace aligator
