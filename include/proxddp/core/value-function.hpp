/**
 * @file    value_function.hpp
 * @brief   Define storage for Q-function and value-function parameters.
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxddp/fwd.hpp"

#include <ostream>

namespace proxddp {

/// @brief  Storage for the value function model parameters.
template <typename _Scalar> struct value_function {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  int ndx_;
  VectorXs Vx_;
  MatrixXs Vxx_;
  Scalar v_;

  value_function(const int ndx) : ndx_(ndx), Vx_(ndx), Vxx_(ndx, ndx) {
    Vx_.setZero();
    Vxx_.setZero();
  }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const value_function &store) {
    oss << "value_function {\n";
    oss << fmt::format("\tndx: {:d}", store.ndx_);
    oss << "\n}";
    return oss;
  }
};

/// @brief   Q-function model parameters
/// @details This struct also provides views for the blocks of interest \f$Q_x,
/// Q_u, Q_y\ldots\f$.
template <typename Scalar> struct q_function {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  int ndx_;
  int nu_;
  int ndy_;
  int ntot = ndx_ + nu_ + ndy_;

  Scalar q_ = 0.;

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

  q_function(const int ndx, const int nu, const int ndy)
      : ndx_(ndx), nu_(nu), ndy_(ndy), grad_(ntot), hess_(ntot, ntot),
        Qx(grad_.head(ndx)), Qu(grad_.segment(ndx, nu)), Qy(grad_.tail(ndy)),
        Qxx(hess_.topLeftCorner(ndx, ndx)), Qxu(hess_.block(0, ndx, ndx, nu)),
        Qxy(hess_.topRightCorner(ndx, ndy)), Quu(hess_.block(ndx, ndx, nu, nu)),
        Quy(hess_.block(ndx, ndx + nu, nu, ndy)),
        Qyy(hess_.bottomRightCorner(ndy, ndy)) {
    grad_.setZero();
    hess_.setZero();
    assert(hess_.rows() == ntot);
    assert(hess_.cols() == ntot);
    assert(grad_.rows() == ntot);
    assert(grad_.cols() == 1);
  }

  friend std::ostream &operator<<(std::ostream &oss, const q_function &store) {
    Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "  [", "]");
    oss << "q_function {\n";
    oss << fmt::format("ndx: {:d}, nu: {:d}, ndy: {:d}", store.ndx_, store.nu_,
                       store.ndy_);
    oss << "\n}";
    return oss;
  }
};

} // namespace proxddp
