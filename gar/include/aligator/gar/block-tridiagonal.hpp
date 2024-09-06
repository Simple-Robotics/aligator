/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "blk-matrix.hpp"
#include "aligator/eigen-macros.hpp"
#include "aligator/third-party/tracy/Tracy.hpp"

namespace aligator {
namespace gar {

template <typename MatrixType>
typename MatrixType::PlainObject
blockTridiagToDenseMatrix(const std::vector<MatrixType> &subdiagonal,
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

template <typename MatrixType, typename InputType, typename OutType,
          typename Scalar = typename MatrixType::Scalar>
bool blockTridiagMatMul(const std::vector<MatrixType> &Asub,
                        const std::vector<MatrixType> &Adiag,
                        const std::vector<MatrixType> &Asuper,
                        const BlkMatrix<InputType, -1, 1> &b,
                        BlkMatrix<OutType, -1, 1> &c, const Scalar beta) {
  ALIGATOR_NOMALLOC_SCOPED;
  const size_t N = Asuper.size();

  c.matrix() *= beta;

  c[0].noalias() += Adiag[0] * b[0];
  c[0].noalias() += Asuper[0] * b[1];

  for (size_t i = 1; i < N; i++) {
    c[i].noalias() += Asub[i - 1] * b[i - 1];
    c[i].noalias() += Adiag[i] * b[i];
    c[i].noalias() += Asuper[i] * b[i + 1];
  }

  c[N].noalias() += Asub[N - 1] * b[N - 1];
  c[N].noalias() += Adiag[N] * b[N];

  return true;
}

/// @brief Solve a symmetric block-tridiagonal \f$Ax=b\f$ problem by in-place
/// factorization. The subdiagonal will be used to store factorization
/// coefficients.
template <typename MatrixType, typename RhsType, typename DecType>
bool symmetricBlockTridiagSolve(std::vector<MatrixType> &subdiagonal,
                                std::vector<MatrixType> &diagonal,
                                const std::vector<MatrixType> &superdiagonal,
                                BlkMatrix<RhsType, -1, 1> &rhs,
                                std::vector<DecType> &facs) {
  ZoneScoped;
  ALIGATOR_NOMALLOC_SCOPED;

  if (subdiagonal.size() != superdiagonal.size() ||
      diagonal.size() != superdiagonal.size() + 1 ||
      rhs.rowDims().size() != diagonal.size()) {
    return false;
  }

  // size of problem
  const size_t N = superdiagonal.size();

  size_t i = N - 1;
  while (true) {
    DecType &ldl = facs[i + 1];
    ldl.compute(diagonal[i + 1]);
    if (ldl.info() != Eigen::Success)
      return false;

    Eigen::Ref<RhsType> r = rhs[i + 1];
    ldl.solveInPlace(r);

    // the math has index of B starting at 1, array starts at 0
    const auto &Bip1 = superdiagonal[i];
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
    const auto &Uip1t = subdiagonal[i];
    rhs[i + 1].noalias() -= Uip1t * rhs[i];
  }

  return true;
}

/// @brief Given the decomposed matrix data, just perform the backward-forward
/// step of the algorithm.
/// @param transposedUfacs - transposed U factors in the decomposition
/// @param superdiagonal - superdiagonal of the original matrix
/// @param diagonalFacs - diagonal factor decompositions
template <typename MatrixType, typename RhsType, typename DecType>
bool blockTridiagRefinementStep(const std::vector<MatrixType> &transposedUfacs,
                                const std::vector<MatrixType> &superdiagonal,
                                const std::vector<DecType> &diagonalFacs,
                                BlkMatrix<RhsType, -1, 1> &rhs) {
  ALIGATOR_NOMALLOC_SCOPED;
  // size of problem
  const size_t N = superdiagonal.size();

  size_t i = N - 1;
  while (true) {
    const DecType &ldl = diagonalFacs[i + 1];
    Eigen::Ref<RhsType> r = rhs[i + 1];
    ldl.solveInPlace(r);

    // the math has index of B starting at 1, array starts at 0
    const auto &Bip1 = superdiagonal[i];

    rhs[i].noalias() -= Bip1 * rhs[i + 1];

    if (i == 0)
      break;
    i--;
  }

  const DecType &ldl = diagonalFacs[0];
  Eigen::Ref<RhsType> r = rhs[0];
  ldl.solveInPlace(r);

  for (size_t i = 0; i < N; i++) {
    const auto &Uip1t = transposedUfacs[i];
    rhs[i + 1].noalias() -= Uip1t * rhs[i];
  }

  return true;
}

/// @copybrief symmetricBlockTridiagSolve(). This version starts by looking down
/// from the top-left corner of the matrix.
template <typename MatrixType, typename RhsType, typename DecType>
bool symmetricBlockTridiagSolveDownLooking(
    std::vector<MatrixType> &subdiagonal, std::vector<MatrixType> &diagonal,
    std::vector<MatrixType> &superdiagonal, BlkMatrix<RhsType, -1, 1> &rhs,
    std::vector<DecType> &facs) {
  ZoneScoped;
  ALIGATOR_NOMALLOC_SCOPED;

  if (subdiagonal.size() != superdiagonal.size() ||
      diagonal.size() != superdiagonal.size() + 1 ||
      rhs.rowDims().size() != diagonal.size()) {
    return false;
  }

  // size of problem
  const size_t N = superdiagonal.size();

  for (size_t i = 0; i < N; i++) {
    DecType &ldl = facs[i];
    ldl.compute(diagonal[i]);
    if (ldl.info() != Eigen::Success)
      return false;

    Eigen::Ref<RhsType> r = rhs[i];
    ldl.solveInPlace(r);

    // the math has index of B starting at 1, array starts at 0
    auto &Bip1 = superdiagonal[i];
    const auto &Cip1 = subdiagonal[i]; // should be B[i+1].transpose()

    rhs[i + 1].noalias() -= Cip1 * rhs[i];
    ldl.solveInPlace(Bip1); // contains L.T = D[i]^-1 * B[i+1]

    // substract B[i+1].T * L.T
    diagonal[i + 1].noalias() -= Cip1 * Bip1;
  }

  {
    DecType &ldl = facs[N];
    ldl.compute(diagonal[N]);
    if (ldl.info() != Eigen::Success)
      return false;
    Eigen::Ref<RhsType> r = rhs[N];
    ldl.solveInPlace(r);
  }

  size_t i = N - 1;
  while (true) {
    const auto &Lim1t = superdiagonal[i];
    rhs[i].noalias() -= Lim1t * rhs[i + 1];

    if (i == 0)
      break;
    i--;
  }
  return true;
}

} // namespace gar
} // namespace aligator
