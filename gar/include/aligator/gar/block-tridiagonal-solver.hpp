#pragma once

#include "aligator/gar/blk-matrix.hpp"
#include "aligator/macros.hpp"
#include "tracy/Tracy.hpp"

namespace aligator {
namespace gar {

template <typename MatrixType>
auto blockTridiagToDenseMatrix(const std::vector<MatrixType> &subdiagonal,
                               const std::vector<MatrixType> &diagonal,
                               const std::vector<MatrixType> &superdiagonal) {
  if (subdiagonal.size() != superdiagonal.size() ||
      diagonal.size() != superdiagonal.size() + 1) {
    throw std::invalid_argument("Wrong lengths");
  }

  using PlainObjectType = typename MatrixType::PlainObject;

  const size_t N = subdiagonal.size();
  Eigen::Index dim = 0;
  for (size_t i = 0; i <= N; i++) {
    dim += diagonal[i].cols();
  }

  PlainObjectType out(dim, dim);
  out.setZero();
  Eigen::Index i0 = 0;
  for (size_t i = 0; i <= N; i++) {
    Eigen::Index d = diagonal[i].cols();
    out.block(i0, i0, d, d) = diagonal[i];

    if (i != N) {
      Eigen::Index dn = superdiagonal[i].cols();
      out.block(i0, i0 + d, d, dn) = superdiagonal[i];
      out.block(i0 + d, i0, dn, d) = subdiagonal[i];
    }

    i0 += d;
  }
  return out;
}

/// Solve a symmetric block-tridiagonal problem by in-place factorization.
/// The subdiagonal will be used to store factorization coefficients.
template <typename MatrixType, typename RhsType, typename DecType>
bool symmetricBlockTridiagSolve(std::vector<MatrixType> &subdiagonal,
                                std::vector<MatrixType> &diagonal,
                                std::vector<MatrixType> &superdiagonal,
                                BlkMatrix<RhsType, -1, 1> &rhs,
                                std::vector<DecType> &facs) {
  ZoneScoped;
  ALIGATOR_NOMALLOC_BEGIN;

  if (subdiagonal.size() != superdiagonal.size() ||
      diagonal.size() != superdiagonal.size() + 1 ||
      rhs.rowDims().size() != diagonal.size()) {
    return false;
  }

  // size of problem
  size_t N = superdiagonal.size();

  size_t i = N - 1;
  while (true) {
    DecType &ldl = facs[i + 1];
    ldl.compute(diagonal[i + 1]);
    if (ldl.info() != Eigen::Success)
      return false;

    Eigen::Ref<RhsType> r = rhs[i + 1];
    ldl.solveInPlace(r);

    // the math has index of B starting at 1, array starts at 0
    auto &Bip1 = superdiagonal[i];
    auto &Cip1 = subdiagonal[i]; // should be Bi.transpose()

    rhs[i].noalias() -= Bip1 * rhs[i + 1];
    ldl.solveInPlace(Cip1); // contains U.T = D[i+1]^-1 * B[i+1].transpose()

    diagonal[i].noalias() -= Bip1 * Cip1;

    if (i == 0)
      break;
    i--;
  }

  {
    DecType &ldl = facs[0];
    ldl.compute(diagonal[0]);
    if (ldl.info() != Eigen::Success)
      return false;
    Eigen::Ref<RhsType> r = rhs[0];
    ldl.solveInPlace(r);
  }

  for (size_t i = 0; i < N; i++) {
    auto &Cip1 = subdiagonal[i];
    rhs[i + 1].noalias() -= Cip1 * rhs[i];
  }

  ALIGATOR_NOMALLOC_END;
  return true;
}

} // namespace gar
} // namespace aligator
