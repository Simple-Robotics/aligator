/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/math.hpp"
#include <array>
#include <numeric>

namespace aligator {

/// @brief Block matrix class, with a fixed-size number of row and column
/// blocks.
template <typename _MatrixType, int _N, int _M = _N> class BlkMatrix {
public:
  using MatrixType = _MatrixType;
  using PlainObject = typename MatrixType::PlainObject;
  using Scalar = typename MatrixType::Scalar;
  enum { N = _N, M = _M, Options = PlainObject::Options };

  using row_dim_t = std::conditional_t<N != -1, std::array<long, size_t(N)>,
                                       std::vector<long>>;
  using col_dim_t = std::conditional_t<M != -1, std::array<long, size_t(M)>,
                                       std::vector<long>>;

  static_assert(N != 0 && M != 0,
                "The BlkMatrix template class only supports nonzero numbers of "
                "blocks in either direction.");

  static_assert(!MatrixType::IsVectorAtCompileTime || (M == 1),
                "Compile-time vector cannot have more than one column block.");

  BlkMatrix()
      : m_data(), m_rowDims(), m_colDims(), m_rowIndices(), m_colIndices(),
        m_totalRows(0), m_totalCols(0) {}

  BlkMatrix(const row_dim_t &rowDims, const col_dim_t &colDims)
      : m_data(), //
        m_rowDims(rowDims), m_colDims(colDims), m_rowIndices(rowDims),
        m_colIndices(colDims), m_totalRows(0), m_totalCols(0) {
    initialize();
  }

  template <typename Other>
  BlkMatrix(const Eigen::MatrixBase<Other> &data, const row_dim_t &rowDims,
            const col_dim_t &colDims)
      : m_data(data.derived()), //
        m_rowDims(rowDims), m_colDims(colDims), m_rowIndices(rowDims),
        m_colIndices(colDims), m_totalRows(0), m_totalCols(0) {
    initialize();
  }

  template <typename Other>
  BlkMatrix(Eigen::MatrixBase<Other> &data, const row_dim_t &rowDims,
            const col_dim_t &colDims)
      : m_data(data.derived()), //
        m_rowDims(rowDims), m_colDims(colDims), m_rowIndices(rowDims),
        m_colIndices(colDims), m_totalRows(0), m_totalCols(0) {
    initialize();
  }

  /// Only-rows constructor (only for vectors)
  template <typename Other>
  BlkMatrix(const Eigen::MatrixBase<Other> &data, const row_dim_t &dims)
      : BlkMatrix(data, dims, {data.cols()}) {}

  /// Only-rows constructor (only for vectors)
  template <typename Other>
  BlkMatrix(Eigen::MatrixBase<Other> &data, const row_dim_t &dims)
      : BlkMatrix(data, dims, {data.cols()}) {}

  /// Only-rows constructor (only for vectors)
  explicit BlkMatrix(const row_dim_t &dims) : BlkMatrix(dims, {1}) {}

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
  }

  void setZero() { m_data.setZero(); }

  MatrixType &matrix() { return m_data; }
  const MatrixType &matrix() const { return m_data; }

  const row_dim_t &rowDims() const { return m_rowDims; }
  const row_dim_t &rowIndices() const { return m_rowIndices; }
  const col_dim_t &colDims() const { return m_colDims; }
  const col_dim_t &colIndices() const { return m_colIndices; }

  long rows() const { return m_totalRows; }
  long cols() const { return m_totalCols; }

  auto topBlkRows(size_t n) {
    using OutType = BlkMatrix<Eigen::Ref<MatrixType>, -1, M>;
    std::vector<long> subRowDims;
    subRowDims.resize(n);
    std::copy_n(m_rowDims.cbegin(), n, subRowDims.begin());
    long ntr = std::accumulate(subRowDims.begin(), subRowDims.end(), 0);
    return OutType(m_data.topRows(ntr), subRowDims, m_colDims);
  }

  template <size_t n> auto topBlkRows() {
    static_assert(n <= N,
                  "Cannot take n block rows of matrix with <n block rows.");
    using RefType = Eigen::Ref<MatrixType>;
    using OutType = BlkMatrix<RefType, n, M>;
    std::array<long, n> subRowDims;
    std::copy_n(m_rowDims.cbegin(), n, subRowDims.begin());
    long ntr = std::accumulate(subRowDims.begin(), subRowDims.end(), 0);
    return OutType(m_data.topRows(ntr), subRowDims, m_colDims);
  }

  friend std::ostream &operator<<(std::ostream &oss, const BlkMatrix &self) {
    return oss << self.m_data;
  }

protected:
  MatrixType m_data;
  row_dim_t m_rowDims;
  col_dim_t m_colDims;
  row_dim_t m_rowIndices;
  col_dim_t m_colIndices;
  long m_totalRows;
  long m_totalCols;

  explicit BlkMatrix(const row_dim_t &dims, std::true_type)
      : BlkMatrix(dims, dims) {}
  explicit BlkMatrix(const row_dim_t &dims, std::false_type)
      : BlkMatrix(dims, {1}) {}

  void initialize() {
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
