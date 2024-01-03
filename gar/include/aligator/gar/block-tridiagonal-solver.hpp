#pragma once

#include "aligator/gar/blk-matrix.hpp"
#include <Eigen/Cholesky>

namespace aligator {
namespace gar {

/// Returns true if block-tridiag matrix data has consistent lengths
namespace internal {
template <typename MatrixType>
bool check_block_tridiag(const std::vector<MatrixType> &subdiagonal,
                         const std::vector<MatrixType> &diagonal,
                         const std::vector<MatrixType> &superdiagonal) {
  return (diagonal.size() == superdiagonal.size() + 1 ||
          diagonal.size() == subdiagonal.size());
}
} // namespace internal

/// Solve a symmetric block-tridiagonal problem by in-place factorization.
/// The subdiagonal will be used to store factorization coefficients.
template <typename MatrixType, typename RhsType>
bool symmetric_block_tridiagonal_solve(std::vector<MatrixType> &subdiagonal,
                                       std::vector<MatrixType> &diagonal,
                                       std::vector<MatrixType> &superdiagonal,
                                       BlkMatrix<RhsType, -1, 1> &rhs) {

  if (!internal::check_block_tridiag(subdiagonal, diagonal, superdiagonal) ||
      rhs.rowDims().size() != diagonal.size()) {
    return false;
  }

  // size of problem
  size_t N = superdiagonal.size();
  using RefType = Eigen::Ref<typename MatrixType::PlainObject>;
  using InPlaceLDLT = Eigen::LDLT<RefType>;

  size_t i = N - 1;
  while (true) {
    InPlaceLDLT ldlt(diagonal[i + 1]);
    if (ldlt.info() != Eigen::Success)
      return false;

    Eigen::Ref<RhsType> r = rhs[i + 1];
    ldlt.solveInPlace(r);

    // the math has index of B starting at 1, array starts at 0
    auto &Bip1 = superdiagonal[i];
    auto &Cip1 = subdiagonal[i]; // should be Bi.transpose()

    rhs[i].noalias() -= Bip1 * rhs[i + 1];
    ldlt.solveInPlace(Cip1); // contains U.T = D[i+1]^-1 * B[i+1].transpose()

    diagonal[i].noalias() -= Bip1 * Cip1;

    if (i == 0)
      break;
    i--;
  }

  {
    InPlaceLDLT ldlt(diagonal[0]);
    if (ldlt.info() != Eigen::Success)
      return false;
    Eigen::Ref<RhsType> r = rhs[0];
    ldlt.solveInPlace(r);
  }

  for (size_t i = 0; i < N; i++) {
    auto &Cip1 = subdiagonal[i];
    rhs[i + 1].noalias() -= Cip1 * rhs[i];
  }

  return true;
}

} // namespace gar
} // namespace aligator
