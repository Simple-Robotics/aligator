/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/math.hpp"
#include <array>
#include <numeric>

namespace aligator {

/// @brief Block matrix class, with a fixed or dynamic-size number of row and
/// column blocks.
/// @tparam Baseline matrix type; the user can use an Eigen::Ref type to create
/// a blocked view to an existing matrix.
/// @tparam N number of block rows.
/// @tparam M number of block columns.
template <typename _MatrixType, int N, int M = N> class BlkMatrix;

template <typename _MatrixType, int _N, int _M> class BlkMatrix {
public:
  using MatrixType = _MatrixType;
  using PlainObject = typename MatrixType::PlainObject;
  using Scalar = typename MatrixType::Scalar;
  using Index = Eigen::Index;
  enum { N = _N, M = _M, Options = PlainObject::Options };
  static constexpr bool IsVectorAtCompileTime =
      MatrixType::IsVectorAtCompileTime;

  using RowDimsType = std::conditional_t<N != -1, std::array<Index, size_t(N)>,
                                         std::vector<Index>>;
  using ColDimsType = std::conditional_t<M != -1, std::array<Index, size_t(M)>,
                                         std::vector<Index>>;

  static_assert(N != 0 && M != 0,
                "The BlkMatrix template class only supports nonzero numbers of "
                "blocks in either direction.");

  static_assert(!IsVectorAtCompileTime || (M == 1),
                "Compile-time vector cannot have more than one column block.");

  BlkMatrix()
      : m_data()
      , m_rowDims()
      , m_colDims()
      , m_rowIndices()
      , m_colIndices()
      , m_totalRows(0)
      , m_totalCols(0) {}

  BlkMatrix(const RowDimsType &rowDims, const ColDimsType &colDims)
      : m_data()
      , m_rowDims(rowDims)
      , m_colDims(colDims)
      , m_rowIndices(rowDims)
      , m_colIndices(colDims)
      , m_totalRows(0)
      , m_totalCols(0) {
    initialize();
  }

  template <typename Other>
  BlkMatrix(const Eigen::MatrixBase<Other> &data, const RowDimsType &rowDims,
            const ColDimsType &colDims)
      : m_data(data.derived())
      , m_rowDims(rowDims)
      , m_colDims(colDims)
      , m_rowIndices(rowDims)
      , m_colIndices(colDims)
      , m_totalRows(0)
      , m_totalCols(0) {
    initialize();
  }

  template <typename Other>
  BlkMatrix(Eigen::MatrixBase<Other> &data, const RowDimsType &rowDims,
            const ColDimsType &colDims)
      : m_data(data.derived())
      , m_rowDims(rowDims)
      , m_colDims(colDims)
      , m_rowIndices(rowDims)
      , m_colIndices(colDims)
      , m_totalRows(0)
      , m_totalCols(0) {
    initialize();
  }

  /// Only-rows constructor (only for vectors)
  template <typename Other>
  BlkMatrix(const Eigen::MatrixBase<Other> &data, const RowDimsType &dims)
      : BlkMatrix(data, dims, {data.cols()}) {}

  /// Only-rows constructor (only for vectors)
  template <typename Other>
  BlkMatrix(Eigen::MatrixBase<Other> &data, const RowDimsType &dims)
      : BlkMatrix(data, dims, {data.cols()}) {}

  operator Eigen::Ref<PlainObject>() { return m_data; }
  operator Eigen::Ref<const PlainObject>() const { return m_data; }

  /// Only-rows constructor (only for vectors)
  explicit BlkMatrix(const RowDimsType &dims)
      : BlkMatrix(dims, {1}) {
    static_assert(IsVectorAtCompileTime,
                  "Constructor only supported for vector types.");
  }

  /// @brief Get the block in position ( @p i, @p j )
  inline auto operator()(size_t i, size_t j) {
    return m_data.block(m_rowIndices[i], m_colIndices[j], m_rowDims[i],
                        m_colDims[j]);
  }

  /// @copybrief operator()
  inline auto operator()(size_t i, size_t j) const {
    return m_data.block(m_rowIndices[i], m_colIndices[j], m_rowDims[i],
                        m_colDims[j]);
  }

  inline auto blockRow(size_t i) {
    return m_data.middleRows(m_rowIndices[i], m_rowDims[i]);
  }

  inline auto blockRow(size_t i) const {
    return m_data.middleRows(m_rowIndices[i], m_rowDims[i]);
  }

  inline auto blockCol(size_t j) const {
    return m_data.middleCols(m_colIndices[j], m_colDims[j]);
  }

  inline auto blockCol(size_t j) {
    return m_data.middleCols(m_colIndices[j], m_colDims[j]);
  }

  auto blockSegment(size_t i) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType);
    return m_data.segment(m_rowIndices[i], m_rowDims[i]);
  }

  auto blockSegment(size_t i) const {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType);
    return m_data.segment(m_rowIndices[i], m_rowDims[i]);
  }

  inline auto operator[](size_t i) { return blockSegment(i); }

  inline auto operator[](size_t i) const { return blockSegment(i); }

  /// Set the data to be equal to some other Eigen object
  template <typename Other>
  BlkMatrix &operator=(const Eigen::MatrixBase<Other> &other) {
    assert(other.rows() == m_data.rows());
    assert(other.cols() == m_data.cols());
    m_data = other;
    return *this;
  }

  void setZero() { m_data.setZero(); }
  static BlkMatrix Zero(const RowDimsType &rowDims,
                        const ColDimsType &colDims) {

    BlkMatrix out(rowDims, colDims);
    out.setZero();
    return out;
  }

  template <typename Other> inline void swap(BlkMatrix<Other, N, M> &other) {
    m_data.swap(other.matrix());
  }

  MatrixType &matrix() { return m_data; }
  const MatrixType &matrix() const { return m_data; }

  template <typename D>
  inline bool isApprox(
      const Eigen::EigenBase<D> &other,
      const Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision()) const {
    return matrix().isApprox(other, prec);
  }

  template <typename D, int N2, int M2>
  inline bool isApprox(
      const BlkMatrix<D, N2, M2> &other,
      const Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision()) const {
    return matrix().isApprox(other.matrix(), prec);
  }

  const RowDimsType &rowDims() const { return m_rowDims; }
  const RowDimsType &rowIndices() const { return m_rowIndices; }
  const ColDimsType &colDims() const { return m_colDims; }
  const ColDimsType &colIndices() const { return m_colIndices; }

  Index rows() const { return m_totalRows; }
  Index cols() const { return m_totalCols; }

  /// @brief Take the top \c n block rows of the block matrix.
  auto topBlkRows(size_t n) {
    using OutType = BlkMatrix<Eigen::Ref<MatrixType>, -1, M>;
    std::vector<Index> subRowDims;
    subRowDims.resize(n);
    std::copy_n(m_rowDims.cbegin(), n, subRowDims.begin());
    Index ntr = std::accumulate(subRowDims.begin(), subRowDims.end(), 0);
    return OutType(m_data.topRows(ntr), subRowDims, m_colDims);
  }

  /// @copybrief topBlkRows().
  /// This version returns a fixed-size block.
  template <size_t n> auto topBlkRows() {
    static_assert(n <= N,
                  "Cannot take n block rows of matrix with <n block rows.");
    using RefType = Eigen::Ref<MatrixType>;
    using OutType = BlkMatrix<RefType, n, M>;
    std::array<Index, n> subRowDims;
    std::copy_n(m_rowDims.cbegin(), n, subRowDims.begin());
    Index ntr = std::accumulate(subRowDims.begin(), subRowDims.end(), 0);
    return OutType(m_data.topRows(ntr), subRowDims, m_colDims);
  }

  friend std::ostream &operator<<(std::ostream &oss, const BlkMatrix &self) {
    return oss << self.m_data;
  }

protected:
  MatrixType m_data;
  RowDimsType m_rowDims;
  ColDimsType m_colDims;
  RowDimsType m_rowIndices;
  ColDimsType m_colIndices;
  Index m_totalRows;
  Index m_totalCols;

  void initialize() {
    m_totalRows = 0;
    m_totalCols = 0;
    for (size_t i = 0; i < m_rowDims.size(); i++) {
      m_rowIndices[i] = m_totalRows;
      m_totalRows += m_rowDims[i];
    }
    for (size_t i = 0; i < m_colDims.size(); i++) {
      m_colIndices[i] = m_totalCols;
      m_totalCols += m_colDims[i];
    }
    m_data.resize(m_totalRows, m_totalCols);
  }
};

} // namespace aligator
