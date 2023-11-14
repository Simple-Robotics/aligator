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

  BlkMatrix(const row_dim_t &rowDims, const col_dim_t &colDims)
      : data(), //
        m_rowDims(rowDims), m_colDims(colDims), m_rowIndices(rowDims),
        m_colIndices(colDims), m_totalRows(0), m_totalCols(0) {
    initialize();
  }

  template <typename Other>
  BlkMatrix(const Eigen::MatrixBase<Other> &data, const row_dim_t &rowDims,
            const col_dim_t &colDims)
      : data(data), //
        m_rowDims(rowDims), m_colDims(colDims), m_rowIndices(rowDims),
        m_colIndices(colDims), m_totalRows(0), m_totalCols(0) {
    initialize();
  }

  explicit BlkMatrix(const row_dim_t &dims)
      : BlkMatrix(dims, std::integral_constant<bool, M != 1>()) {}

  /// @brief Get the block in position ( @p i, @p j )
  inline auto operator()(size_t i, size_t j) {
    return data.block(m_rowIndices[i], m_colIndices[j], m_rowDims[i],
                      m_colDims[j]);
  }

  /// @copybrief operator()
  inline auto operator()(size_t i, size_t j) const {
    return data.block(m_rowIndices[i], m_colIndices[j], m_rowDims[i],
                      m_colDims[j]);
  }

  inline auto blockRow(size_t i) {
    return data.middleRows(m_rowIndices[i], m_rowDims[i]);
  }

  inline auto blockRow(size_t i) const {
    return data.middleRows(m_rowIndices[i], m_rowDims[i]);
  }

  auto blockSegment(size_t i) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType);
    return data.segment(m_rowIndices[i], m_rowDims[i]);
  }

  auto blockSegment(size_t i) const {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType);
    return data.segment(m_rowIndices[i], m_rowDims[i]);
  }

  void setZero() { data.setZero(); }

  MatrixType data;

  const row_dim_t &rowDims() const { return m_rowDims; }
  const row_dim_t &rowIndices() const { return m_rowIndices; }
  const col_dim_t &colDims() const { return m_colDims; }
  const col_dim_t &colIndices() const { return m_colIndices; }

  long rows() const { return m_totalRows; }
  long cols() const { return m_totalCols; }

  friend std::ostream &operator<<(std::ostream &oss, const BlkMatrix &self) {
    return oss << self.data;
  }

protected:
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
    data.resize(m_totalRows, m_totalCols);
  }
};

template <class T, int N, int M>
auto topBlkRows(size_t n, BlkMatrix<T, N, M> &mat) {
  using RefType = Eigen::Ref<T>;
  using OutType = BlkMatrix<RefType, -1, M>;
  const auto &rowDims = mat.rowDims();
  const auto &colDims = mat.colDims();
  std::vector<long> subRowDims;
  subRowDims.resize(n);
  std::copy_n(rowDims.cbegin(), n, subRowDims.begin());
  auto ntr = std::accumulate(subRowDims.begin(), subRowDims.end(), 0);
  return OutType(mat.data.topRows(ntr), subRowDims, colDims);
}

template <size_t n, class T, int N, int M>
auto topBlkRows(BlkMatrix<T, N, M> &mat) {
  static_assert(n <= N,
                "Cannot take n block rows of matrix with <n block rows.");
  using RefType = Eigen::Ref<T>;
  using OutType = BlkMatrix<RefType, n, M>;
  const auto &rowDims = mat.rowDims();
  const auto &colDims = mat.colDims();
  std::array<long, n> subRowDims;
  std::copy_n(rowDims.cbegin(), n, subRowDims.begin());
  auto ntr = std::accumulate(subRowDims.begin(), subRowDims.end(), 0);
  return OutType(mat.data.topRows(ntr), subRowDims, colDims);
}

} // namespace aligator
