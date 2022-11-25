/**
 * @file    value_storage.hpp
 * @brief   Define storage for Q-function and value-function parameters.
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxddp/fwd.hpp"

#include <ostream>

namespace proxddp {
namespace internal {

/// @brief  Contiguous storage for the value function parameters.
///
/// @details This provides storage for the matrix \f[
///     \begin{bmatrix} 2v & V_x^\top \\ V_x & V_{xx} \end{bmatrix}
/// \f]
template <typename _Scalar> struct value_storage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  int ndx_;
  VectorXs Vx_;
  MatrixXs Vxx_;
  Scalar v_;

  value_storage(const int ndx) : ndx_(ndx), Vx_(ndx), Vxx_(ndx, ndx) {
    Vx_.setZero();
    Vxx_.setZero();
  }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const value_storage &store) {
    Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "  [", "]");
    oss << "value_storage {\n";
    oss << fmt::format("\tndx: {:d}", store.ndx_);
    oss << "\n}";
    return oss;
  }
};

/** @brief  Contiguous storage for Q-function parameters with corresponding
 *          sub-matrix views.
 *
 * @details  The storage layout is as follows:
 * \f[
 *    \begin{bmatrix}
 *      2q    & Q_x^\top  & Q_u^top & Q_y^\top  \\
 *      Q_x   & Q_{xx}    & Q_{xu}  & Q_{xy}    \\
 *      Q_u   & Q_{ux}    & Q_{uu}  & Q_{uy}    \\
 *      Q_y   & Q_{yx}    & Q_{yu}  & Q_{yy}
 *    \end{bmatrix}
 * ]\f
 */
template <typename Scalar> struct q_storage {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  int ndx_;
  int nu_;
  int ndy_;
  int ntot = ndx_ + nu_ + ndy_;

  Scalar q_2_ = 0.;

  Scalar &q_2() { return q_2_; }

  VectorXs grad_;
  MatrixXs hess_;

  VectorRef Qx;
  VectorRef Qu;
  VectorRef Qy;

  MatrixRef Qxx;
  MatrixRef Qxu;
  MatrixRef Qxy;
  MatrixRef Quu;
  MatrixRef Quy;
  MatrixRef Qyy;

  q_storage(const int ndx, const int nu, const int ndy)
      : ndx_(ndx), nu_(nu), ndy_(ndy), grad_(ntot), hess_(ntot, ntot),
        Qx(grad_.head(ndx)), Qu(grad_.segment(ndx, nu)), Qy(grad_.tail(ndy)),
        Qxx(hess_.topLeftCorner(ndx, ndx)), Qxu(hess_.block(0, ndx, ndx, nu)),
        Qxy(hess_.topRightCorner(ndx, ndy)), Quu(hess_.block(ndx, ndx, nu, nu)),
        Quy(hess_.block(ndx, ndx + nu, nu, ndy)),
        Qyy(hess_.bottomRightCorner(ndy, ndy)) {
    assert(hess_.rows() == ntot);
    assert(hess_.cols() == ntot);
    assert(grad_.rows() == ntot);
    assert(grad_.cols() == 1);
  }

  friend std::ostream &operator<<(std::ostream &oss, const q_storage &store) {
    Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "  [", "]");
    oss << "q_storage {\n";
    oss << fmt::format("ndx: {:d}, nu: {:d}, ndy: {:d}", store.ndx_, store.nu_,
                       store.ndy_);
    oss << "\n}";
    return oss;
  }
};

} // namespace internal
} // namespace proxddp
