#pragma once

#include "aligator/math.hpp"
#include <array>

namespace aligator {

/// Block matrix class, with a fixed-size number of row and column blocks.
template <typename _MatrixType, size_t _N, size_t _M = _N> class BlkMatrix {
public:
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  enum { N = _N, M = _M, Options = MatrixType::Options };

  static_assert(N > 0 && M > 0,
                "The BlkMatrix template class only supports nonzero numbers of "
                "blocks in either direction.");

  static_assert(!MatrixType::IsVectorAtCompileTime || (M == 1),
                "Compile-time vector cannot have more than one column block.");

  BlkMatrix(const std::array<long, N> &rowDims,
            const std::array<long, M> &colDims)
      : data(), m_rowDims(rowDims), m_colDims(colDims), m_totalRows(0),
        m_totalCols(0) {
    for (size_t i = 0; i < N; i++) {
      m_rowIndices[i] = m_totalRows;
      m_totalRows += rowDims[i];
    }
    for (size_t i = 0; i < M; i++) {
      m_colIndices[i] = m_totalCols;
      m_totalCols += colDims[i];
    }
    data.resize(m_totalRows, m_totalCols);
  }

  BlkMatrix(const std::array<long, N> &dims) : BlkMatrix(dims, dims) {}

  inline auto operator()(size_t i, size_t j) {
    return data.block(m_rowIndices[i], m_colIndices[j], m_rowDims[i],
                      m_colDims[j]);
  }

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

  const std::array<long, N> &rowDims() const { return m_rowDims; }
  const std::array<long, M> &colDims() const { return m_colDims; }

protected:
  std::array<long, N> m_rowDims;
  std::array<long, M> m_colDims;
  std::array<long, N> m_rowIndices;
  std::array<long, M> m_colIndices;
  long m_totalRows;
  long m_totalCols;
};

} // namespace aligator
