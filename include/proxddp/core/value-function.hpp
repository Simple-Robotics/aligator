/**
 * @file    ValueFunctionTpl.hpp
 * @brief   Define storage for Q-function and value-function parameters.
 * @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
 */
#pragma once

#include "proxddp/fwd.hpp"

#include <ostream>

namespace proxddp {

/// @brief  Storage for the value function model parameters.
template <typename _Scalar> struct ValueFunctionTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  int ndx_;
  VectorXs Vx_;
  MatrixXs Vxx_;
  Scalar v_;

  ValueFunctionTpl(const int ndx) : ndx_(ndx), Vx_(ndx), Vxx_(ndx, ndx) {
    Vx_.setZero();
    Vxx_.setZero();
  }

  bool operator==(const ValueFunctionTpl &other) {
    return (ndx_ == other.ndx_) && Vx_.isApprox(other.Vx_) &&
           Vxx_.isApprox(other.Vxx_) && math::scalar_close(v_, other.v_);
  }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const ValueFunctionTpl &store) {
    oss << "ValueFunction {\n";
    oss << fmt::format("\tndx: {:d}", store.ndx_);
    oss << "\n}";
    return oss;
  }
};

/// @brief   Q-function model parameters
/// @details This struct also provides views for the blocks of interest \f$Q_x,
/// Q_u, Q_y\ldots\f$.
template <typename _Scalar> struct QFunctionTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  long ndx_;
  long nu_;
  long ndy_;
  long ntot() const { return ndx_ + nu_ + ndy_; }

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

  QFunctionTpl(const long ndx, const long nu, const long ndy)
      : ndx_(ndx), nu_(nu), ndy_(ndy), grad_(ntot()), hess_(ntot(), ntot()),
        Qx(grad_.head(ndx)), Qu(grad_.segment(ndx, nu)), Qy(grad_.tail(ndy)),
        Qxx(hess_.topLeftCorner(ndx, ndx)), Qxu(hess_.block(0, ndx, ndx, nu)),
        Qxy(hess_.topRightCorner(ndx, ndy)), Quu(hess_.block(ndx, ndx, nu, nu)),
        Quy(hess_.block(ndx, ndx + nu, nu, ndy)),
        Qyy(hess_.bottomRightCorner(ndy, ndy)) {
    grad_.setZero();
    hess_.setZero();
    assert(hess_.rows() == ntot());
    assert(hess_.cols() == ntot());
    assert(grad_.rows() == ntot());
    assert(grad_.cols() == 1);
  }

  bool operator==(const QFunctionTpl &) { return false; }
  QFunctionTpl(const QFunctionTpl &qf)
      : QFunctionTpl(qf.ndx_, qf.nu_, qf.ndy_) {
    ndx_ = qf.ndx_;
    nu_ = qf.nu_;
    ndy_ = qf.ndy_;
    q_ = qf.q_;
    grad_ = qf.grad_;
    hess_ = qf.hess_;

    redef_refs(*this);
  }

  QFunctionTpl(QFunctionTpl &&qf) : QFunctionTpl(0, 0, 0) { swap(*this, qf); }

  QFunctionTpl &operator=(QFunctionTpl &&qf) {
    swap(*this, qf);
    return *this;
  }

  QFunctionTpl &operator=(const QFunctionTpl &qf) {
    ndx_ = qf.ndx_;
    nu_ = qf.nu_;
    ndy_ = qf.ndy_;
    q_ = qf.q_;
    grad_ = qf.grad_;
    hess_ = qf.hess_;

    redef_refs(*this);
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const QFunctionTpl &store) {
    oss << "QFunction {\n";
    oss << fmt::format("ndx: {:d}, nu: {:d}, ndy: {:d}", store.ndx_, store.nu_,
                       store.ndy_);
    oss << "\n}";
    return oss;
  }

  friend void swap(QFunctionTpl &qa, QFunctionTpl &qb) {
    using std::swap;

    swap(qa.ndx_, qb.ndx_);
    swap(qa.nu_, qb.nu_);
    swap(qa.ndy_, qb.ndy_);
    swap(qa.q_, qb.q_);
    swap(qa.grad_, qb.grad_);
    swap(qa.hess_, qb.hess_);

    redef_refs(qa);
    redef_refs(qb);
  }

protected:
  // reinitialize Ref members
  static void redef_refs(QFunctionTpl &q) {
    auto ndx = q.ndx_;
    auto nu = q.nu_;
    auto ndy = q.ndy_;
    new (&q.Qx) VectorRef(q.grad_.head(ndx));
    new (&q.Qu) VectorRef(q.grad_.segment(ndx, nu));
    new (&q.Qy) VectorRef(q.grad_.tail(ndy));

    new (&q.Qxx) MatrixRef(q.hess_.topLeftCorner(ndx, ndx));
    new (&q.Qxu) MatrixRef(q.hess_.block(0, ndx, ndx, nu));
    new (&q.Qxy) MatrixRef(q.hess_.topRightCorner(ndx, ndy));
    new (&q.Quu) MatrixRef(q.hess_.block(ndx, ndx, nu, nu));
    new (&q.Quy) MatrixRef(q.hess_.block(ndx, ndx + nu, nu, ndy));
    new (&q.Qyy) MatrixRef(q.hess_.bottomRightCorner(ndy, ndy));
  }
};

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/value-function.txx"
#endif
