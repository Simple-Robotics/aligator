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
  int ndx;
  MatrixXs storage;
  Scalar &v_2() { return storage.coeffRef(0, 0); }

  value_storage(const int ndx)
      : ndx(ndx), storage(MatrixXs::Zero(ndx + 1, ndx + 1)) {}

  decltype(auto) Vx() { return storage.col(0).tail(ndx); }
  decltype(auto) Vx() const { return storage.col(0).tail(ndx); }
  decltype(auto) Vxx() { return storage.bottomRightCorner(ndx, ndx); }
  decltype(auto) Vxx() const { return storage.bottomRightCorner(ndx, ndx); }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const value_storage &store) {
    Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "  [", "]");
    oss << "value_storage {\n";
    oss << store.storage.format(CleanFmt);
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

  int ntot;
  MatrixXs storage;

  Scalar &q_2() { return storage.coeffRef(0, 0); }

  VectorRef grad_;
  MatrixRef hess_;

  VectorRef Qx;
  VectorRef Qu;
  VectorRef Qy;

  MatrixRef Qxx;
  MatrixRef Qxu;
  MatrixRef Qxy;
  MatrixRef Quu;
  MatrixRef Quy;
  MatrixRef Qyy;

  q_storage(const int ndx1, const int nu, const int ndx2)
      : ntot(ndx1 + nu + ndx2), storage(MatrixXs::Zero(ntot + 1, ntot + 1)),
        grad_(storage.col(0).tail(ntot)),
        hess_(storage.bottomRightCorner(ntot, ntot)), Qx(grad_.head(ndx1)),
        Qu(grad_.segment(ndx1, nu)), Qy(grad_.tail(ndx2)),
        Qxx(hess_.topLeftCorner(ndx1, ndx1)),
        Qxu(hess_.block(0, ndx1, ndx1, nu)),
        Qxy(hess_.topRightCorner(ndx1, ndx2)),
        Quu(hess_.block(ndx1, ndx1, nu, nu)),
        Quy(hess_.block(ndx1, ndx1 + nu, nu, ndx2)),
        Qyy(hess_.bottomRightCorner(ndx2, ndx2)) {
    assert(hess_.rows() == ntot);
    assert(hess_.cols() == ntot);
    assert(grad_.rows() == ntot);
    assert(grad_.cols() == 1);
  }
};

} // namespace internal
} // namespace proxddp
