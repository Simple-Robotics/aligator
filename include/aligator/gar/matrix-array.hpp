/// @copyright Copyright (C) 2024 INRIA
#pragma once

#include "aligator/memory/eigen-map.hpp"

namespace aligator {

/// @brief An array of Eigen::Matrix objects managed by a unique memory
/// allocator/memory pool.
template <typename Scalar, bool IsVector, int Options = Eigen::ColMajor>
class MatrixArray {
public:
  using allocator_type = polymorphic_allocator;
  static constexpr int Alignment = Eigen::AlignedMax;
  using IndexType = Eigen::Index;
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic,
                                   IsVector ? 1 : Eigen::Dynamic, Options>;
  using StoredType = Eigen::Map<MatrixType, Alignment>;
  using ConstStoredType = Eigen::Map<const MatrixType, Alignment>;

  MatrixArray(const std::pmr::vector<IndexType> &rows,
              const std::pmr::vector<IndexType> &cols,
              allocator_type alloc = {})
      : storage(alloc), m_rows(rows, alloc), m_cols(cols, alloc) {
    static_assert(!IsVector, "Constructor disallowed for vector types.");
    auto N = m_rows.size();
    assert(N == m_cols.size());
    storage.reserve(N);
    for (size_t i = 0; i < N; i++) {
      storage.emplace_back(
          allocate_eigen_map<MatrixType>(alloc, m_rows[i], m_cols[i]));
    }
  }

  /// @brief Constructor, for vector types.
  MatrixArray(const std::pmr::vector<IndexType> &dims,
              allocator_type alloc = {})
      : storage(alloc), m_rows(dims, alloc) {
    static_assert(IsVector, "Constructor only allowed for vector types.");
    auto N = m_rows.size();
    storage.reserve(N);
    for (size_t i = 0; i < N; i++) {
      storage.emplace_back(allocate_eigen_map<MatrixType>(alloc, m_rows[i]));
    }
  }

  ConstStoredType operator[](size_t i) const { return storage[i]; }
  StoredType operator[](size_t i) { return storage[i]; }

  ConstStoredType at(size_t i) const { return storage.at(i); }
  StoredType at(size_t i) { return storage.at(i); }

  MatrixArray(const MatrixArray &other, allocator_type alloc = {})
      : MatrixArray(other.m_rows, other.m_cols, alloc) {
    this->storage = other.storage;
  }

  MatrixArray(MatrixArray &&other)
      : storage(other.storage, other.get_allocator()),
        m_rows(std::move(other.m_rows)), m_cols(std::move(other.m_cols)) {
    other.storage.clear();
  }

  ~MatrixArray() {
    auto alloc = get_allocator();
    for (size_t i = 0; i < storage.size(); i++) {
      deallocate_map(storage[i], alloc);
    }
  }

  allocator_type get_allocator() const { return storage.get_allocator(); }

private:
  std::pmr::vector<StoredType> storage;
  std::pmr::vector<IndexType> m_rows, m_cols;
};

template <typename Scalar> using VectorArray = MatrixArray<Scalar, true>;

} // namespace aligator
