#pragma once
/// @file costs.hpp
/// @brief Define cost functions.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxddp/fwd.hpp"
#include "proxddp/core/clone.hpp"

namespace proxddp {
/** @brief Stage costs \f$ \ell(x, u) \f$ for control problems.
 */
template <typename _Scalar> struct CostAbstractTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using CostData = CostDataAbstractTpl<Scalar>;

  /// @brief State dimension
  const int ndx;
  /// @brief Control dimension
  const int nu;

  CostAbstractTpl(const int ndx, const int nu) : ndx(ndx), nu(nu) {}

  /// @brief Evaluate the cost function.
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const = 0;

  /// @brief Compute the cost gradients \f$(\ell_x, \ell_u)\f$
  virtual void computeGradients(const ConstVectorRef &x,
                                const ConstVectorRef &u,
                                CostData &data) const = 0;

  /// @brief Compute the cost Hessians \f$(\ell_{ij})_{i,j \in \{x,u\}}\f$
  virtual void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                               CostData &data) const = 0;

  virtual shared_ptr<CostData> createData() const {
    return std::make_shared<CostData>(ndx, nu);
  }

  virtual ~CostAbstractTpl() = default;
};

/// @brief  Data struct for CostAbstractTpl
template <typename _Scalar> struct CostDataAbstractTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  int ndx_, nu_;

  Scalar value_;
  VectorXs grad_;
  MatrixXs hess_;

  /// @brief Gradient \f$\ell_x\f$
  VectorRef Lx_;
  /// @brief Gradient \f$\ell_u\f$
  VectorRef Lu_;
  /// @brief Hessian \f$\ell_{xx}\f$
  MatrixRef Lxx_;
  /// @brief Hessian \f$\ell_{xu}\f$
  MatrixRef Lxu_;
  /// @brief Hessian \f$\ell_{ux}\f$
  MatrixRef Lux_;
  /// @brief Hessian \f$\ell_{uu}\f$
  MatrixRef Luu_;

  CostDataAbstractTpl(const int ndx, const int nu)
      : ndx_(ndx), nu_(nu), value_(0.), grad_(ndx + nu),
        hess_(ndx + nu, ndx + nu), Lx_(grad_.head(ndx)), Lu_(grad_.tail(nu)),
        Lxx_(hess_.topLeftCorner(ndx, ndx)),
        Lxu_(hess_.topRightCorner(ndx, nu)),
        Lux_(hess_.bottomLeftCorner(nu, ndx)),
        Luu_(hess_.bottomRightCorner(nu, nu)) {
    grad_.setZero();
    hess_.setZero();
  }
};

} // namespace proxddp
