/// @copyright Copyright (C) 2025 INRIA
#pragma once

#include "aligator/memory/allocator.hpp"
#include <Eigen/Core>

namespace aligator {

/// @brief Thin wrapper around Eigen::Map representing a matrix object with
/// memory managed by a C++17 polymorphic allocator.
///
/// This is an allocator-aware (AA) type, meaning it contains a \c
/// allocator_type typedef.
///
/// We follow the std::pmr::polymorphic_allocator style, thus the STL traits
/// \c propagate_on_container_copy_assignment, \c
/// propagate_on_container_move_assignment, and \c propagate_on_container_swap
/// all evaluate to false. The object's allocator does not change during its
/// lifetime.
template <typename _Scalar, int _Rows, int _Cols,
          int _Options = Eigen::ColMajor, int Alignment = Eigen::AlignedMax>
class ManagedMatrix {
  using alloc_traits = std::allocator_traits<polymorphic_allocator>;
  static_assert((_Rows == Eigen::Dynamic) || (_Cols == Eigen::Dynamic));

  void allocate() {
    m_data = m_allocator.allocate<_Scalar>(size_t(size()), Alignment);
    m_allocated_size = size();
  }

  void deallocate() {
    if (m_data)
      m_allocator.deallocate<_Scalar>(m_data, size_t(m_allocated_size),
                                      Alignment);
  }

public:
  using Scalar = _Scalar;
  static constexpr int Rows = _Rows;
  static constexpr int Cols = _Cols;
  static constexpr int Options = _Options;
  using MatrixType = Eigen::Matrix<Scalar, Rows, Cols, Options>;
  using MapType = Eigen::Map<MatrixType, Alignment>;
  using ConstMapType = Eigen::Map<const MatrixType, Alignment>;
  using Index = Eigen::Index;
  using allocator_type = polymorphic_allocator;

  static constexpr bool IsVectorAtCompileTime =
      MatrixType::IsVectorAtCompileTime;

  /// Extended default constructor.
  explicit ManagedMatrix(const allocator_type &allocator = {})
      : m_allocated_size(0)
      , m_rows()
      , m_cols()
      , m_allocator(allocator) {}

  explicit ManagedMatrix(Index size, const allocator_type &allocator)
      : m_rows()
      , m_cols()
      , m_allocator(allocator) {
    static_assert(IsVectorAtCompileTime);
    if constexpr (Rows == 1) {
      m_rows = 1;
      m_cols = size;
    } else if constexpr (Cols == 1) {
      m_rows = size;
      m_cols = 1;
    }
    this->allocate();
    assert(m_data);
  }

  explicit ManagedMatrix(Index rows, Index cols,
                         const allocator_type &allocator)
      : m_rows(rows)
      , m_cols(cols)
      , m_allocator(allocator) {
    this->allocate();
    assert(m_data);
  }

  ManagedMatrix(const ManagedMatrix &other, const allocator_type &alloc = {})
      : ManagedMatrix(other.rows(), other.cols(), alloc) {
    this->to_map() = other.to_const_map();
  }

  /// Extended move constructor, will use the provided allocator.
  ManagedMatrix(ManagedMatrix &&other, const allocator_type &alloc)
      : ManagedMatrix(alloc) {
    m_rows = std::move(other.m_rows);
    m_cols = std::move(other.m_cols);

    if (!other.m_data)
      return;

    if (m_allocator == other.get_allocator()) {
      // same allocator: just steal the pointer
      m_data = other.m_data;
      other.m_data = nullptr;
    } else if (size() > 0) {
      // different allocator: allocate, copy values
      this->allocate();
      this->to_map() = other.to_const_map();
    }
    // dtor of 'other' cleans up if necessary
  }

  /// Nonextended move constructor, will use _moved-from_ object's allocator.
  ManagedMatrix(ManagedMatrix &&other) noexcept
      : m_allocator(other.get_allocator()) {
    m_allocated_size = other.m_allocated_size;
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_data = other.m_data;

    other.m_data = nullptr;
    other.m_allocated_size = 0;
  }

  /// \brief Copy constructor from another Eigen matrix.
  template <typename Derived>
  ManagedMatrix(const Eigen::MatrixBase<Derived> &mat,
                const allocator_type &alloc = {})
      : ManagedMatrix(mat.rows(), mat.cols(), alloc) {
    this->to_map() = mat;
  }

  ManagedMatrix &operator=(const ManagedMatrix &other) {
    // set new dimensions
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    if (m_allocated_size < other.size()) {
      // reallocate
      this->deallocate();
      this->allocate();
      assert(m_allocated_size == size());
    }
    this->to_map() = other.to_const_map();
    return *this;
  }

  ManagedMatrix &operator=(ManagedMatrix &&other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;

    if (!other.m_data)
      return *this;

    if (m_allocator == other.get_allocator()) {
      // just steal the pointer, return early
      m_data = other.m_data;
      other.m_data = nullptr;
      return *this;
    } else {
      this->resize(other.rows(), other.cols());
    }
    this->to_map() = other.to_const_map();
    return *this;
  }

  template <typename Derived>
  ManagedMatrix &operator=(const Eigen::MatrixBase<Derived> &mat) {
    this->resize(mat.rows(), mat.cols());
    this->to_map() = mat;
    return *this;
  }

  void resize(Index size) {
    static_assert(IsVectorAtCompileTime);
    if constexpr (Rows == 1) {
      m_rows = 1;
      m_cols = size;
    } else if constexpr (Cols == 1) {
      m_rows = size;
      m_cols = 1;
    }
    if (m_allocated_size < size) {
      this->deallocate();
      this->allocate();
    }
  }

  void resize(Index rows, Index cols) {
    const Index new_size = rows * cols;
    m_rows = rows;
    m_cols = cols;
    if (m_allocated_size < new_size) {
      this->deallocate();
      this->allocate();
    }
  }

  Index rows() const noexcept { return m_rows; }

  Index cols() const noexcept { return m_cols; }

  Index size() const noexcept { return m_rows * m_cols; }

  /// Get current allocated size.
  Index allocated_size() const noexcept { return m_allocated_size; }

  /// @brief Accessor to retrieve the allocator used for this matrix.
  [[nodiscard]] allocator_type get_allocator() const noexcept {
    return m_allocator;
  }

  [[nodiscard]] explicit operator MapType() {
    if constexpr (IsVectorAtCompileTime) {
      return MapType{m_data, size()};
    } else {
      return MapType{m_data, m_rows, m_cols};
    }
  }

  [[nodiscard]] explicit operator ConstMapType() const {
    if constexpr (IsVectorAtCompileTime) {
      return ConstMapType{m_data, size()};
    } else {
      return ConstMapType{m_data, m_rows, m_cols};
    }
  }

  ~ManagedMatrix() { this->deallocate(); }

  /// \brief Obtain mutable map.
  MapType to_map() { return MapType{*this}; }

  /// \brief Obtain const map since this is const.
  ConstMapType to_map() const { return to_const_map(); }

  /// \brief Obtain a const map.
  ConstMapType to_const_map() const { return ConstMapType{*this}; }

  void setZero() { this->to_map().setZero(); }

  void setZero(Index rows, Index cols) {
    this->resize(rows, cols);
    this->setZero();
  }

  void setZero(Index size) {
    this->resize(size);
    this->setZero();
  }

  void setRandom() { this->to_map().setRandom(); }

  void setIdentity() { this->to_map().setIdentity(); }

  void setConstant(Scalar s) { this->to_map().setConstant(s); }

  auto noalias() { return to_map().noalias(); }

  auto diagonal() { return to_map().diagonal(); }
  auto diagonal() const { return to_const_map().diagonal(); }

  auto topRows(Index n) { return to_map().topRows(n); }
  auto topRows(Index n) const { return to_const_map().topRows(n); }

  auto head(Index n) { return to_map().head(n); }
  auto head(Index n) const { return to_const_map().head(n); }

  auto tail(Index n) { return to_map().tail(n); }
  auto tail(Index n) const { return to_const_map().tail(n); }

  bool isApprox(const ManagedMatrix &other,
                Scalar prec = std::numeric_limits<Scalar>::epsilon()) const {
    return to_const_map().isApprox(other.to_const_map(), prec);
  }

  template <typename Derived>
  bool isApprox(const Eigen::DenseBase<Derived> &mat,
                Scalar prec = std::numeric_limits<Scalar>::epsilon()) const {
    return to_const_map().isApprox(mat, prec);
  }

  /// \brief Pointer to stored data.
  [[nodiscard]] Scalar *data() { return m_data; }

  /// \copybrief data().
  [[nodiscard]] const Scalar *data() const { return m_data; }

private:
  Index m_allocated_size; // might be different than size()
  Index m_rows;
  Index m_cols;
  Scalar *m_data{nullptr};
  allocator_type m_allocator;
};

} // namespace aligator
