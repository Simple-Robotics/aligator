/// @copyright Copyright (C) 2024-2025 INRIA
#pragma once

#include "aligator/math.hpp"
#include "aligator/memory/allocator.hpp"

namespace aligator {

/// A replacement for `Eigen::Matrix` but compatible with C++17 polymorphic
/// allocators (through the aligator::polymorphic_allocator subclass which
/// extends the API).
///
/// This is implemented a subclass of `Eigen::Map`.
///
/// The implementation is inspired from the `arena_matrix` class template in
/// Stan's math library: See:
/// https://mc-stan.org/math/classstan_1_1math_1_1arena__matrix_3_01_matrix_type_00_01require__eigen__dense__base__t_3_01_matrix_type_01_4_01_4.html
///
/// \tparam MatrixType The original plain Eigen matrix type.
/// \tparam Alignment The desired alignment for the data pointer.
template <typename MatrixType, int Alignment = Eigen::AlignedMax>
class ArenaMatrix;

template <typename MatrixType, int Alignment>
class ArenaMatrix : public Eigen::Map<std::decay_t<MatrixType>, Alignment> {
public:
  using Scalar = typename std::decay_t<MatrixType>::Scalar;
  using PlainObject = std::decay_t<MatrixType>;
  using Base = Eigen::Map<PlainObject, Alignment>;
  using Index = Eigen::Index;
  using ConstMapType = Eigen::Map<const PlainObject, Alignment>;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  using allocator_type = polymorphic_allocator;
  using Base::data;

  ArenaMatrix()
      : Base::Map(nullptr,
                  RowsAtCompileTime == Eigen::Dynamic ? 0 : RowsAtCompileTime,
                  ColsAtCompileTime == Eigen::Dynamic ? 0 : ColsAtCompileTime)
      , m_allocator()
      , m_alloc_size(0l) {}

  explicit ArenaMatrix(const allocator_type &allocator)
      : Base::Map(nullptr,
                  RowsAtCompileTime == Eigen::Dynamic ? 0 : RowsAtCompileTime,
                  ColsAtCompileTime == Eigen::Dynamic ? 0 : ColsAtCompileTime)
      , m_allocator(allocator)
      , m_alloc_size(0l) {}

  [[nodiscard]] allocator_type get_allocator() const noexcept {
    return m_allocator;
  }

  ArenaMatrix(Index cols, Index rows, const allocator_type &allocator = {})
      : Base::Map(_allocate(cols * rows, allocator), cols, rows)
      , m_allocator(allocator)
      , m_alloc_size(cols * rows) {}

  explicit ArenaMatrix(Index size, const allocator_type &allocator = {})
      : Base::Map(_allocate(size, allocator), size)
      , m_allocator(allocator)
      , m_alloc_size(size) {}

  template <EigenMatrix D>
  ArenaMatrix(const D &other, const allocator_type &alloc = {})
      : Base::Map(_allocate(other.size(), alloc), get_rows(other),
                  get_cols(other))
      , m_allocator(alloc)
      , m_alloc_size(other.size()) {
    Base::operator=(other);
  }

  explicit ArenaMatrix(const ArenaMatrix &other,
                       const allocator_type &alloc = {})
      : ArenaMatrix(other.rows(), other.cols(), alloc) {
    // copy values from other, delegate to base type's operator=
    Base::operator=(other);
  }

  ArenaMatrix(ArenaMatrix &&other) noexcept
      : Base::Map(other.data(), get_rows(other), get_cols(other))
      , m_allocator(other.m_allocator)
      , m_alloc_size(other.m_alloc_size) {
    other.m_data = nullptr;
    other.m_alloc_size = 0;
  }

  /// Extended move constructor.
  ArenaMatrix(ArenaMatrix &&other, const allocator_type &alloc)
      : ArenaMatrix(alloc) {
    if (m_allocator == other.m_allocator) {
      // do not reallocate
      new (this) Base(other.data(), get_rows(other), get_cols(other));
      m_alloc_size = other.m_alloc_size;
    } else {
      // allocate
      this->resize(get_rows(other), get_cols(other));
      Base::operator=(other);
    }
    other.m_data = nullptr;
    other.m_alloc_size = 0;
  }

  template <EigenMatrix D> ArenaMatrix &operator=(const D &other) {
    this->resize(get_rows(other), get_cols(other));
    Base::operator=(other);
    return *this;
  }

  /// Explicitly define copy assignment operator.
  /// Required because polymorphic_allocator has its copy assignment operator
  /// implicitly deleted.
  ArenaMatrix &operator=(const ArenaMatrix &other) {
    this->resize(get_rows(other), get_cols(other));
    Base::operator=(other);
    return *this;
  }

  using Base::operator=;

  ArenaMatrix &operator=(ArenaMatrix &&other) {
    if (this == &other)
      return *this;

    if (m_allocator == other.m_allocator) {
      this->deallocate();
      m_alloc_size = other.m_alloc_size;
      new (this) Base(other.data(), get_rows(other), get_cols(other));
    } else {
      this->resize(get_rows(other), get_cols(other));
      Base::operator=(other);
    }
    other.m_data = nullptr;
    return *this;
  }

  operator ConstMapType() const {
    return ConstMapType(this->m_data, this->rows(), this->cols());
  }

  ~ArenaMatrix() { this->deallocate(); }

  /// Resize this matrix. This will reallocate.
  void resize(Index size) {
    if (m_alloc_size != size) {
      this->deallocate();
      Scalar *p = _allocate(size, m_allocator);
      m_alloc_size = size;
      new (this) Base(p, size);
    } else {
      // simply replace the Map object
      new (this) Base(this->m_data, size);
    }
  }

  /// @copydoc resize().
  void resize(Index rows, Index cols) {
    const Index size = rows * cols;
    if (m_alloc_size != size) {
      this->deallocate();
      Scalar *p = _allocate(size, m_allocator);
      m_alloc_size = size;
      new (this) Base(p, rows, cols);
    } else {
      // simply replace the Map object
      new (this) Base(this->m_data, rows, cols);
    }
  }

  /// @brief
  void conservativeResize(Index size) {
    if (m_alloc_size < size) {
      Scalar *p = _allocate(size, m_allocator);
      std::copy_n(this->m_data, m_alloc_size, p);
      this->deallocate();
      m_alloc_size = size;
      new (this) Base(p, size);
    } else {
      new (this) Base(this->m_data, size);
    }
  }

  void conservativeResize(Index rows, Index cols) {
    const Index size = rows * cols;
    if (m_alloc_size < size) {
      Scalar *p = _allocate(size, m_allocator);
      std::copy_n(this->m_data, m_alloc_size, p);
      this->deallocate();
      m_alloc_size = size;
      new (this) Base(p, rows, cols);
    } else {
      new (this) Base(this->m_data, rows, cols);
    }
  }

  /// @brief Current allocated size for this matrix. This is what allows
  /// resizing without reallocation.
  Index allocatedSize() const noexcept { return m_alloc_size; }

  using Base::setZero;

  ArenaMatrix &setZero(Index newSize) {
    *this = MatrixType::Zero(newSize);
    return *this;
  }

  ArenaMatrix &setZero(Index rows, Index cols) {
    *this = MatrixType::Zero(rows, cols);
    return *this;
  }

  using Base::setIdentity;

  ArenaMatrix &setIdentity(Index rows, Index cols) {
    *this = MatrixType::Identity(rows, cols);
  }

private:
  template <typename T> constexpr auto get_rows(const T &x) {
    return (RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1) ||
                   (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)
               ? x.cols()
               : x.rows();
  }
  template <typename T> constexpr auto get_cols(const T &x) {
    return (RowsAtCompileTime == 1 && T::ColsAtCompileTime == 1) ||
                   (ColsAtCompileTime == 1 && T::RowsAtCompileTime == 1)
               ? x.rows()
               : x.cols();
  }

  [[nodiscard]] static auto *_allocate(Index size, allocator_type alloc) {
    return alloc.allocate<Scalar>(size_t(size), Alignment);
  }

  void deallocate() {
    if (this->m_data)
      m_allocator.deallocate(this->m_data, size_t(this->size()), Alignment);
  }

private:
  allocator_type m_allocator;
  Index m_alloc_size;
};

} // namespace aligator

namespace Eigen {
namespace internal {

// lifted from the stan-dev/math library
template <typename T> struct traits<aligator::ArenaMatrix<T>> {
  using base = traits<Eigen::Map<T>>;
  using Scalar = typename base::Scalar;
  using XprKind = typename Eigen::internal::traits<std::decay_t<T>>::XprKind;
  using StorageKind =
      typename Eigen::internal::traits<std::decay_t<T>>::StorageKind;
  static constexpr int RowsAtCompileTime =
      Eigen::internal::traits<std::decay_t<T>>::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime =
      Eigen::internal::traits<std::decay_t<T>>::ColsAtCompileTime;
  enum {
    PlainObjectTypeInnerSize = base::PlainObjectTypeInnerSize,
    InnerStrideAtCompileTime = base::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = base::OuterStrideAtCompileTime,
    Alignment = base::Alignment,
    Flags = base::Flags
  };
};

} // namespace internal
} // namespace Eigen
