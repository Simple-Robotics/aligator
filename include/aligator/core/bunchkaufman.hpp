/// @file
/// @author Sarah El-Kazdadi
/// @copyright Copyright (C) 2023
#pragma once

#include "Eigen/Core"

namespace Eigen {
template <typename MatrixType_, int UpLo_ = Lower> struct BunchKaufman;

namespace internal {
template <typename MatrixType_, int UpLo_>
struct traits<BunchKaufman<MatrixType_, UpLo_>> : traits<MatrixType_> {
  typedef MatrixXpr XprKind;
  typedef SolverStorage StorageKind;
  typedef int StorageIndex;
  enum { Flags = 0 };
};

template <typename MatrixType, typename IndicesType>
ComputationInfo bunch_kaufman_in_place_unblocked(MatrixType &a,
                                                 IndicesType &pivots,
                                                 Index &pivot_count) {
  using Scalar = typename MatrixType::Scalar;
  using Real = typename Eigen::NumTraits<Scalar>::Real;
  const Real alpha = (Real(1) + numext::sqrt(Real(17))) / Real(8);
  using std::swap;

  pivot_count = 0;

  const Index n = a.rows();
  if (n == 0) {
    return Success;
  } else if (n == 1) {
    if (numext::abs(numext::real(a(0, 0))) == Real(0)) {
      return NumericalIssue;
    } else {
      a(0, 0) = Real(1) / numext::real(a(0, 0));
      return Success;
    }
  }

  Index k = 0;
  while (k < n) {
    Index k_step = 1;
    Real abs_akk = numext::abs(numext::real(a(k, k)));
    Index imax = 0;
    Real colmax = Real(0);

    if (k + 1 < n) {
      colmax = a.col(k).segment(k + 1, n - k - 1).cwiseAbs().maxCoeff(&imax);
    }
    imax += k + 1;

    Index kp;
    if (numext::maxi(abs_akk, colmax) == Real(0)) {
      return NumericalIssue;
    } else {
      if (abs_akk >= colmax * alpha) {
        kp = k;
      } else {
        Real rowmax = Real(0);
        if (imax - k > 0) {
          rowmax = a.row(imax).segment(k, imax - k).cwiseAbs().maxCoeff();
        }
        if (n - imax - 1 > 0) {
          rowmax = numext::maxi(rowmax, a.col(imax)
                                            .segment(imax + 1, n - imax - 1)
                                            .cwiseAbs()
                                            .maxCoeff());
        }

        if (abs_akk >= (alpha * colmax) * (colmax / rowmax)) {
          kp = k;
        } else if (numext::abs(numext::real(a(imax, imax))) >= alpha * rowmax) {
          kp = imax;
        } else {
          kp = imax;
          k_step = 2;
        }
      }

      Index kk = k + k_step - 1;
      if (kp != kk) {
        pivot_count += 1;
        a.col(kk)
            .segment(kp + 1, n - kp - 1)
            .swap(a.col(kp).segment(kp + 1, n - kp - 1));

        for (Index j = kk + 1; j < kp; ++j) {
          Scalar tmp = a(j, kk);
          a(j, kk) = numext::conj(a(kp, j));
          a(kp, j) = numext::conj(tmp);
        }
        a(kp, kk) = numext::conj(a(kp, kk));
        swap(a(kk, kk), a(kp, kp));

        if (k_step == 2) {
          swap(a(k + 1, k), a(kp, k));
        }
      }

      if (k_step == 1) {
        Real d11 = Real(1) / numext::real(a(k, k));
        a(k, k) = Scalar(d11);

        auto x = a.middleRows(k + 1, n - k - 1).col(k);
        auto trailing =
            a.middleRows(k + 1, n - k - 1).middleCols(k + 1, n - k - 1);

        for (Index j = 0; j < n - k - 1; ++j) {
          Scalar d11xj = numext::conj(x(j)) * d11;
          for (Index i = j; i < n - k - 1; ++i) {
            Scalar xi = x(i);
            trailing(i, j) -= d11xj * xi;
          }
          trailing(j, j) = Scalar(numext::real(trailing(j, j)));
        }

        x *= d11;
      } else {
        Real d21_abs = numext::abs(a(k + 1, k));
        Real d21_inv = Real(1) / d21_abs;
        Real d11 = d21_inv * numext::real(a(k + 1, k + 1));
        Real d22 = d21_inv * numext::real(a(k, k));

        Real t = Real(1) / ((d11 * d22) - Real(1));
        Real d = t * d21_inv;
        Scalar d21 = a(k + 1, k) * d21_inv;

        a(k, k) = Scalar(d11 * d);
        a(k + 1, k) = -d21 * d;
        a(k + 1, k + 1) = Scalar(d22 * d);

        for (Index j = k + 2; j < n; ++j) {
          Scalar wk = ((a(j, k) * d11) - (a(j, k + 1) * d21)) * d;
          Scalar wkp1 =
              ((a(j, k + 1) * d22) - (a(j, k) * numext::conj(d21))) * d;

          for (Index i = j; i < n; ++i) {
            a(i, j) -=
                a(i, k) * numext::conj(wk) + a(i, k + 1) * numext::conj(wkp1);
          }
          a(j, j) = Scalar(numext::real(a(j, j)));

          a(j, k) = wk;
          a(j, k + 1) = wkp1;
        }
      }
    }

    // pivots matrix store int by default
    // Like Eigen, we assume that we will never
    // have a matrix that will have more rows/columns
    // than int can old
    if (k_step == 1) {
      pivots[k] = static_cast<int>(kp);
    } else {
      pivots[k] = static_cast<int>(-1 - kp);
      pivots[k + 1] = static_cast<int>(-1 - kp);
    }

    k += k_step;
  }

  return Success;
}

template <typename MatrixType, typename WType, typename IndicesType>
ComputationInfo
bunch_kaufman_in_place_one_block(MatrixType &a, WType &w, IndicesType &pivots,
                                 Index &pivot_count, Index &processed_cols) {
  using Scalar = typename MatrixType::Scalar;
  using Real = typename Eigen::NumTraits<Scalar>::Real;

  pivot_count = 0;
  processed_cols = 0;
  Real alpha = (Real(1) + numext::sqrt(Real(17))) / Real(8);

  Index n = a.rows();
  if (n == 0) {
    return Success;
  }

  Index nb = w.cols();
  Index k = 0;
  while (k < n && k + 1 < nb) {
    w.col(k).segment(k, n - k) = a.col(k).segment(k, n - k);
    {
      auto w_row = w.row(k).segment(0, k);
      auto w_col = w.col(k).segment(k, n - k);
      w_col.noalias() -= a.block(k, 0, n - k, k) * w_row.transpose();
    }
    w(k, k) = Scalar(numext::real(w(k, k)));

    Index k_step = 1;
    Real abs_akk = numext::abs(numext::real(w(k, k)));
    Index imax = 0;
    Real colmax = Real(0);

    if (k + 1 < n) {
      colmax = w.col(k).segment(k + 1, n - k - 1).cwiseAbs().maxCoeff(&imax);
    }
    imax += k + 1;

    Index kp;
    if (numext::maxi(abs_akk, colmax) == Real(0)) {
      return NumericalIssue;
    } else {
      if (abs_akk >= colmax * alpha) {
        kp = k;
      } else {
        w.col(k + 1).segment(k, imax - k) =
            a.row(imax).segment(k, imax - k).adjoint();
        w.col(k + 1).segment(imax, n - imax) =
            a.col(imax).segment(imax, n - imax);

        {
          auto w_row = w.row(imax).segment(0, k);
          auto w_col = w.col(k + 1).segment(k, n - k);
          w_col.noalias() -= a.block(k, 0, n - k, k) * w_row.transpose();
        }
        w(imax, k + 1) = Scalar(numext::real(w(imax, k + 1)));

        Real rowmax = Real(0);
        if (imax - k > 0) {
          rowmax = w.col(k + 1).segment(k, imax - k).cwiseAbs().maxCoeff();
        }
        if (n - imax - 1 > 0) {
          rowmax = numext::maxi(rowmax, w.col(k + 1)
                                            .segment(imax + 1, n - imax - 1)
                                            .cwiseAbs()
                                            .maxCoeff());
        }

        if (abs_akk >= (alpha * colmax) * (colmax / rowmax)) {
          kp = k;
        } else if (numext::abs(numext::real(w(imax, k + 1))) >=
                   alpha * rowmax) {
          kp = imax;
          w.col(k).segment(k, n - k) = w.col(k + 1).segment(k, n - k);
        } else {
          kp = imax;
          k_step = 2;
        }
      }

      Index kk = k + k_step - 1;
      if (kp != kk) {
        pivot_count += 1;

        a(kp, kp) = a(kk, kk);
        for (Index j = kk + 1; j < kp; ++j) {
          a(kp, j) = numext::conj(a(j, kk));
        }
        a.col(kp).segment(kp + 1, n - kp - 1) =
            a.col(kk).segment(kp + 1, n - kp - 1);
        a.row(kk).segment(0, k).swap(a.row(kp).segment(0, k));
        w.row(kk).segment(0, kk + 1).swap(w.row(kp).segment(0, kk + 1));
      }

      if (k_step == 1) {
        a.col(k).segment(k, n - k) = w.col(k).segment(k, n - k);

        Real d11 = Real(1) / numext::real(w(k, k));
        a(k, k) = Scalar(d11);
        auto x = a.middleRows(k + 1, n - k - 1).col(k);
        x *= d11;
        w.col(k).segment(k + 1, n - k - 1) =
            w.col(k).segment(k + 1, n - k - 1).conjugate();
      } else {
        Real d21_abs = numext::abs(w(k + 1, k));
        Real d21_inv = Real(1) / d21_abs;
        Real d11 = d21_inv * numext::real(w(k + 1, k + 1));
        Real d22 = d21_inv * numext::real(w(k, k));

        Real t = Real(1) / ((d11 * d22) - Real(1));
        Scalar d21 = w(k + 1, k) * d21_inv;
        Real d = t * d21_inv;

        a(k, k) = Scalar(d11 * d);
        a(k + 1, k) = -d21 * d;
        a(k + 1, k + 1) = Scalar(d22 * d);

        for (Index j = k + 2; j < n; ++j) {
          Scalar wk = ((w(j, k) * d11) - (w(j, k + 1) * d21)) * d;
          Scalar wkp1 =
              ((w(j, k + 1) * d22) - (w(j, k) * numext::conj(d21))) * d;

          a(j, k) = wk;
          a(j, k + 1) = wkp1;
        }

        w.col(k).segment(k + 1, n - k - 1) =
            w.col(k).segment(k + 1, n - k - 1).conjugate();
        w.col(k + 1).segment(k + 2, n - k - 2) =
            w.col(k + 1).segment(k + 2, n - k - 2).conjugate();
      }
    }

    // pivots matrix store int by default
    // Like Eigen, we assume that we will never
    // have a matrix that will have more rows/columns
    // than int can old
    if (k_step == 1) {
      pivots[k] = static_cast<int>(kp);
    } else {
      pivots[k] = static_cast<int>(-1 - kp);
      pivots[k + 1] = static_cast<int>(-1 - kp);
    }

    k += k_step;
  }

  auto a_left = a.bottomRows(n - k).leftCols(k);
  auto a_right = a.bottomRows(n - k).rightCols(n - k);

  a_right.template triangularView<Lower>() -=
      a_left * w.block(k, 0, n - k, k).transpose();
  Index j = k - 1;
  processed_cols = k;

  while (true) {
    Index jj = j;
    Index jp = pivots[j];
    if (jp < 0) {
      jp = -1 - jp;
      j -= 1;
    }

    if (j == 0) {
      return Success;
    }
    j -= 1;
    if (jp != jj) {
      a.row(jp).segment(0, j + 1).swap(a.row(jj).segment(0, j + 1));
    }
    if (j == 0) {
      return Success;
    }
  }
}

template <typename MatrixType, typename VecType, typename IndicesType,
          typename WorkspaceType>
ComputationInfo bunch_kaufman_in_place(MatrixType &a, VecType &subdiag,
                                       IndicesType &pivots, WorkspaceType &w,
                                       Index &pivot_count) {
  Index n = a.rows();

  const Index blocksize = w.cols();

  Index k = 0;
  while (k < n) {
    Index kb = 0;
    Index k_pivot_count = 0;
    auto a_block = a.block(k, k, n - k, n - k);
    auto pivots_block = pivots.segment(k, n - k);
    ComputationInfo info = InvalidInput;
    if (blocksize != 0 && blocksize < n - k) {
      info = internal::bunch_kaufman_in_place_one_block(
          a_block, w, pivots_block, k_pivot_count, kb);
    } else {
      info = internal::bunch_kaufman_in_place_unblocked(a_block, pivots_block,
                                                        k_pivot_count);
      kb = n - k;
    }
    if (info != Success) {
      return info;
    }

    for (Index j = k; j < k + kb; ++j) {
      auto &p = pivots.coeffRef(j);
      // pivots matrix store int by default
      // Like Eigen, we assume that we will never
      // have a matrix that will have more rows/columns
      // than int can old
      if (p >= 0) {
        p += static_cast<int>(k);
      } else {
        p -= static_cast<int>(k);
      }
    }

    pivot_count += k_pivot_count;
    k += kb;
  }

  using Scalar = typename MatrixType::Scalar;

  k = 0;
  while (k < n) {
    if (pivots[k] < 0) {
      subdiag(k) = a(k + 1, k);
      subdiag(k + 1) = Scalar(0);
      a(k + 1, k) = Scalar(0);
      k += 2;
    } else {
      subdiag(k) = Scalar(0);
      k += 1;
    }
  }

  k = 0;
  while (k < n) {
    Index p = pivots[k];
    if (p < 0) {
      p = -1 - p;
      a.row(k + 1).segment(0, k).swap(a.row(p).segment(0, k));
      k += 2;
    } else {
      a.row(k).segment(0, k).swap(a.row(p).segment(0, k));
      k += 1;
    }
  }

  return Success;
}

template <typename MatrixType, bool Conjugate> struct BK_Traits;

template <typename MatrixType> struct BK_Traits<MatrixType, false> {
  typedef TriangularView<const MatrixType, UnitLower> MatrixL;
  typedef TriangularView<const typename MatrixType::AdjointReturnType,
                         UnitUpper>
      MatrixU;
  static inline MatrixL getL(const MatrixType &m) { return MatrixL(m); }
  static inline MatrixU getU(const MatrixType &m) {
    return MatrixU(m.adjoint());
  }
};

template <typename MatrixType> struct BK_Traits<MatrixType, true> {
  typedef typename MatrixType::ConjugateReturnType ConjugateReturnType;
  typedef TriangularView<const ConjugateReturnType, UnitLower> MatrixL;
  typedef TriangularView<const typename MatrixType::TransposeReturnType,
                         UnitUpper>
      MatrixU;
  static inline MatrixL getL(const MatrixType &m) {
    return MatrixL(m.conjugate());
  }
  static inline MatrixU getU(const MatrixType &m) {
    return MatrixU(m.transpose());
  }
};

template <bool Conjugate, typename MatrixType, typename VecType,
          typename IndicesType, typename Rhs>
void bunch_kaufman_solve_in_place(MatrixType const &L, VecType const &subdiag,
                                  IndicesType const &pivots, Rhs &x) {
  Index n = L.rows();

  Index k;

  k = 0;
  while (k < n) {
    Index p = pivots(k);
    if (p < 0) {
      p = -1 - p;
      x.row(k + 1).swap(x.row(p));
      k += 2;
    } else {
      x.row(k).swap(x.row(p));
      k += 1;
    }
  }

  using Traits = BK_Traits<MatrixType, Conjugate>;

  Traits::getL(L).solveInPlace(x);

  k = 0;
  while (k < n) {
    Index p = pivots(k);
    if (p < 0) {
      using Scalar = typename MatrixType::Scalar;
      using Real = typename Eigen::NumTraits<Scalar>::Real;

      Scalar akp1k = subdiag(k);
      Real ak = numext::real(L(k, k));
      Real akp1 = numext::real(L(k + 1, k + 1));

      if (Conjugate) {
        akp1k = numext::conj(akp1k);
      }

      for (Index j = 0; j < x.cols(); ++j) {
        Scalar xk = x(k, j);
        Scalar xkp1 = x(k + 1, j);

        x(k, j) = xk * ak + xkp1 * numext::conj(akp1k);
        x(k + 1, j) = xkp1 * akp1 + xk * akp1k;
      }

      k += 2;
    } else {
      x.row(k) *= numext::real(L(k, k));
      k += 1;
    }
  }

  Traits::getU(L).solveInPlace(x);

  k = n;
  while (k > 0) {
    k -= 1;
    Index p = pivots(k);
    if (p < 0) {
      p = -1 - p;
      x.row(k).swap(x.row(p));
      k -= 1;
    } else {
      x.row(k).swap(x.row(p));
    }
  }
}
} // namespace internal

template <typename MatrixType_, int UpLo_>
struct BunchKaufman : SolverBase<BunchKaufman<MatrixType_, UpLo_>> {
  enum {
    MaxRowsAtCompileTime = MatrixType_::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType_::MaxColsAtCompileTime,
    UpLo = UpLo_
  };
  using MatrixType = MatrixType_;
  using PlainObject = typename MatrixType::PlainObject;
  using Base = SolverBase<BunchKaufman>;
  static constexpr Index BlockSize = 32;
  EIGEN_GENERIC_PUBLIC_INTERFACE(BunchKaufman)
  using VecType = Matrix<Scalar, RowsAtCompileTime, 1, Eigen::DontAlign,
                         MaxRowsAtCompileTime>;
  friend class SolverBase<BunchKaufman>;

  using IndicesType =
      typename Transpositions<RowsAtCompileTime,
                              MaxRowsAtCompileTime>::IndicesType;
  using PermutationType =
      PermutationMatrix<RowsAtCompileTime, MaxRowsAtCompileTime>;

  BunchKaufman()
      : m_matrix()
      , m_subdiag()
      , m_pivot_count(0)
      , m_pivots()
      , m_isInitialized(false)
      , m_info(ComputationInfo::InvalidInput)
      , m_blocksize()
      , m_workspace() {}
  explicit BunchKaufman(Index size)
      : m_matrix(size, size)
      , m_subdiag(size)
      , m_pivot_count(0)
      , m_pivots(size)
      , m_isInitialized(false)
      , m_info(ComputationInfo::InvalidInput)
      , m_blocksize(size <= BlockSize ? 0 : BlockSize)
      , m_workspace(size, m_blocksize) {}

  template <typename InputType>
  explicit BunchKaufman(const EigenBase<InputType> &matrix)
      : m_matrix(matrix.rows(), matrix.cols())
      , m_subdiag(matrix.rows())
      , m_pivot_count(0)
      , m_pivots(matrix.rows())
      , m_isInitialized(false)
      , m_info(ComputationInfo::InvalidInput)
      , m_blocksize(matrix.rows() <= BlockSize ? 0 : BlockSize)
      , m_workspace(matrix.rows(), m_blocksize) {
    this->compute(matrix.derived());
  }

  // EIGEN_DEVICE_FUNC inline EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT
  // { return m_matrix.rows(); }
  EIGEN_DEVICE_FUNC inline Index rows() const EIGEN_NOEXCEPT {
    return m_matrix.rows();
  }
  // EIGEN_DEVICE_FUNC inline EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT
  // { return m_matrix.cols(); }
  EIGEN_DEVICE_FUNC inline Index cols() const EIGEN_NOEXCEPT {
    return m_matrix.cols();
  }

  inline const MatrixType &matrixLDLT() const { return m_matrix; }

  inline const IndicesType &pivots() const { return m_pivots; }

  inline const VecType &subdiag() const { return m_subdiag; }

  template <typename InputType>
  BunchKaufman &compute(const EigenBase<InputType> &matrix);

  ComputationInfo info() const { return m_info; }

#ifdef EIGEN_PARSED_BY_DOXYGEN
  template <typename Rhs>
  inline const Solve<LDLT, Rhs> solve(const MatrixBase<Rhs> &b) const;
#endif

#ifndef EIGEN_PARSED_BY_DOXYGEN
  template <typename RhsType, typename DstType>
  void _solve_impl(const RhsType &rhs, DstType &dst) const;

  template <bool Conjugate, typename RhsType, typename DstType>
  void _solve_impl_transposed(const RhsType &rhs, DstType &dst) const;

  template <typename RhsType>
  bool solveInPlace(Eigen::MatrixBase<RhsType> &bAndX) const;
#endif

private:
  MatrixType m_matrix;
  VecType m_subdiag;
  Index m_pivot_count;
  IndicesType m_pivots;
  bool m_isInitialized;
  ComputationInfo m_info;
  Index m_blocksize;
  PlainObject m_workspace;
};

#ifndef EIGEN_PARSED_BY_DOXYGEN
template <typename MatrixType_, int UpLo_>
template <typename RhsType, typename DstType>
void BunchKaufman<MatrixType_, UpLo_>::_solve_impl(const RhsType &rhs,
                                                   DstType &dst) const {
  dst = rhs;
  internal::bunch_kaufman_solve_in_place<false>(this->m_matrix, this->m_subdiag,
                                                this->m_pivots, dst);
}

template <typename MatrixType_, int UpLo_>
template <bool Conjugate, typename RhsType, typename DstType>
void BunchKaufman<MatrixType_, UpLo_>::_solve_impl_transposed(
    const RhsType &rhs, DstType &dst) const {
  dst = rhs;
  internal::bunch_kaufman_solve_in_place<!Conjugate>(
      this->m_matrix, this->m_subdiag, this->m_pivots, dst);
}

template <typename MatrixType_, int UpLo_>
template <typename RhsType>
bool BunchKaufman<MatrixType_, UpLo_>::solveInPlace(
    Eigen::MatrixBase<RhsType> &bAndX) const {
  bAndX = this->solve(bAndX);

  return true;
}
#endif

template <typename MatrixType_, int UpLo_>
template <typename InputType>
BunchKaufman<MatrixType_, UpLo_> &
BunchKaufman<MatrixType_, UpLo_>::compute(const EigenBase<InputType> &a) {
  eigen_assert(a.rows() == a.cols());
  Index n = a.rows();
  this->m_matrix.resize(n, n);
  this->m_subdiag.resize(n);
  this->m_pivots.resize(n);

  this->m_matrix.setZero();
  this->m_subdiag.setZero();
  this->m_pivots.setZero();
  this->m_blocksize = n <= BlockSize ? 0 : BlockSize;
  this->m_workspace.setZero(n, this->m_blocksize);

  this->m_matrix.template triangularView<Lower>() =
      a.derived().template triangularView<UpLo_>();
  this->m_info = internal::bunch_kaufman_in_place(
      this->m_matrix, this->m_subdiag, this->m_pivots, this->m_workspace,
      this->m_pivot_count);
  this->m_isInitialized = true;
  return *this;
}
} // namespace Eigen

namespace aligator {
using Eigen::BunchKaufman;
}
