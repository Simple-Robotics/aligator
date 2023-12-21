/// @file costs.hpp
/// @brief Define cost functions.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/clone.hpp"
#include <proxsuite-nlp/manifold-base.hpp>

namespace aligator {
/** @brief Stage costs \f$ \ell(x, u) \f$ for control problems.
 */
template <typename _Scalar> struct CostAbstractTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using CostData = CostDataAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  /// @brief State dimension
  shared_ptr<Manifold> space;
  /// @brief Control dimension
  int nu;

  int nx() const { return space->nx(); }
  int ndx() const { return space->ndx(); }

  CostAbstractTpl(shared_ptr<Manifold> space, const int nu)
      : space(space), nu(nu) {}

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
    return std::make_shared<CostData>(ndx(), nu);
  }

  virtual ~CostAbstractTpl() = default;
};

/// @brief  Data struct for CostAbstractTpl
template <typename _Scalar> struct CostDataAbstractTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
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

  CostDataAbstractTpl(const CostAbstractTpl<Scalar> &cost)
      : CostDataAbstractTpl(cost.ndx(), cost.nu) {}

  virtual ~CostDataAbstractTpl() = default;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/cost-abstract.txx"
#endif
