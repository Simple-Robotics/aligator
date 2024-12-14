#pragma once

#include "memory-allocator.hpp"
#include <new>

namespace aligator {

template <typename MatrixType, int Alignment = Eigen::AlignedMax>
auto allocate_eigen_map(polymorphic_allocator &alloc, Eigen::Index rows,
                        Eigen::Index cols) {
  using MapType = Eigen::Map<MatrixType, Alignment>;
  using Scalar = typename MatrixType::Scalar;
  size_t size = size_t(rows * cols);
  Scalar *data = alloc.allocate<Scalar>(size, Alignment);
  return MapType{data, rows, cols};
}

template <typename MatrixType, int Alignment = Eigen::AlignedMax>
auto allocate_eigen_map(polymorphic_allocator &alloc, Eigen::Index size) {
  using MapType = Eigen::Map<MatrixType, Alignment>;
  using Scalar = typename MatrixType::Scalar;
  Scalar *data = alloc.allocate<Scalar>(size_t(size), Alignment);
  return MapType{data, size};
}

/// @brief In-place construct a map from another one by stealing the other's
/// data.
template <typename MatrixType, int Alignment>
void emplace_map_steal(Eigen::Map<MatrixType, Alignment> &map,
                       Eigen::Map<MatrixType, Alignment> &other) {
  EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(MatrixType);
  using MapType = Eigen::Map<MatrixType, Alignment>;
  typename MatrixType::Scalar *data = other.data();
  if (data) {
    if constexpr (MatrixType::IsVectorAtCompileTime) {
      new (&map) MapType{data, other.size()};
    } else {
      new (&map) MapType{data, other.rows(), other.cols()};
    }
  }
  other.~MapType();
}

/// @brief Use `placement new` to create an Eigen::Map object with given
/// dimensions and data pointer.
template <typename MatrixType, int Alignment>
void emplace_map_from_data(Eigen::Map<MatrixType, Alignment> &map,
                           Eigen::Index rows, Eigen::Index cols,
                           typename MatrixType::Scalar *data) {
  EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(MatrixType);
  using MapType = Eigen::Map<MatrixType, Alignment>;
  new (&map) MapType{data, rows, cols};
}

/// @copybrief emplace_map()
template <typename MatrixType, int Alignment>
void emplace_map_from_data(Eigen::Map<MatrixType, Alignment> &map,
                           Eigen::Index size,
                           typename MatrixType::Scalar *data) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType);
  EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(MatrixType);
  using MapType = Eigen::Map<MatrixType, Alignment>;
  new (&map) MapType{data, size};
}

/// @brief Use `placement new` and an allocator to create an Eigen::Map object
/// to it.
template <typename MatrixType, int Alignment>
void emplace_allocated_map(Eigen::Map<MatrixType, Alignment> &map,
                           Eigen::Index rows, Eigen::Index cols,
                           polymorphic_allocator &alloc) {
  using Scalar = typename MatrixType::Scalar;
  Scalar *data = alloc.template allocate<Scalar>(size_t(rows * cols));
  emplace_map_from_data(map, rows, cols, data);
}

/// @copybrief emplace_allocated_map()
template <typename MatrixType, int Alignment>
void emplace_allocated_map(Eigen::Map<MatrixType, Alignment> &map,
                           Eigen::Index size, polymorphic_allocator &alloc) {
  using Scalar = typename MatrixType::Scalar;
  Scalar *data = alloc.template allocate<Scalar>(size_t(size), Alignment);
  emplace_map_from_data(map, size, data);
}

template <typename MatrixType, int Alignment>
void emplace_resize_map(Eigen::Map<MatrixType, Alignment> &map,
                        Eigen::Index rows, Eigen::Index cols,
                        polymorphic_allocator &alloc) {
  EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(MatrixType);
  using MapType = Eigen::Map<MatrixType, Alignment>;
  using Scalar = typename MatrixType::Scalar;
  bool need_reallocate = map.size() != rows * cols;
  Scalar *data = map.data();
  if (data && need_reallocate) {
    alloc.template deallocate<Scalar>(data, map.size(), Alignment);
    data = alloc.template allocate<Scalar>(size_t(rows * cols));
  }
  map.~MapType();
  new (&map) MapType{data, rows, cols};
}

template <typename MatrixType, int Alignment>
void deallocate_map(Eigen::Map<MatrixType, Alignment> &map,
                    polymorphic_allocator &alloc) {
  EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(MatrixType);
  using Scalar = typename MatrixType::Scalar;
  size_t dealloc_size = size_t(map.size());
  if (map.data() != NULL)
    alloc.template deallocate<Scalar>(map.data(), dealloc_size, Alignment);
}

/// @brief Create a deep copy of a managed Eigen::Map object.
template <typename MatrixType, int Alignment>
void emplace_map_copy(Eigen::Map<MatrixType, Alignment> &map,
                      const Eigen::Map<MatrixType, Alignment> &other,
                      polymorphic_allocator &alloc) {
  if constexpr (MatrixType::IsVectorAtCompileTime) {
    emplace_allocated_map(map, other.size(), alloc);
  } else {
    emplace_allocated_map(map, other.rows(), other.cols(), alloc);
  }
  // now copy values using Eigen's copy operator=
  map = other;
}

} // namespace aligator
