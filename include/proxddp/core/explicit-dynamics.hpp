/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/dynamics.hpp"

#include <proxnlp/manifold-base.hpp>

#include <fmt/format.h>

namespace proxddp {

/** @brief Explicit forward dynamics model \f$ x_{k+1} = f(x_k, u_k) \f$.
 *
 *  @details    Forward dynamics \f$ x_{k+1} = f(x_k, u_k) \f$.
 *              The corresponding residuals for multiple-shooting formulations
 * are \f[ \bar{f}(x_k, u_k, x_{k+1}) = f(x_k, u_k) \ominus x_{k+1}. \f]
 */
template <typename _Scalar>
struct ExplicitDynamicsModelTpl : DynamicsModelTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = DynamicsModelTpl<Scalar>;
  using BaseData = DynamicsDataTpl<Scalar>;
  using Data = ExplicitDynamicsDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;

  bool is_explicit() const { return true; }

  /// Constructor requires providing the next state's manifold.
  ExplicitDynamicsModelTpl(ManifoldPtr next_state, const int nu);
  virtual ~ExplicitDynamicsModelTpl() = default;

  /// @brief Evaluate the forward discrete dynamics.
  void virtual forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       Data &data) const = 0;

  /// @brief Compute the Jacobians of the forward dynamics.
  void virtual dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const = 0;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const;

  virtual shared_ptr<BaseData> createData() const;
};

/// @brief    Specific data struct for explicit dynamics
/// ExplicitDynamicsModelTpl.
template <typename _Scalar>
struct ExplicitDynamicsDataTpl : FunctionDataTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = FunctionDataTpl<Scalar>;
  using Base::Ju_;
  using Base::Jx_;
  using Base::Jy_;
  using Base::value_;
  VectorXs xnext_;
  VectorXs dx_;
  MatrixXs Jtmp_xnext;

  VectorRef xnext_ref;
  VectorRef dx_ref;

  ExplicitDynamicsDataTpl(const int ndx1, const int nu, const int nx2,
                          const int ndx2);
  virtual ~ExplicitDynamicsDataTpl() = default;

  template <typename S>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const ExplicitDynamicsDataTpl<S> &self);
};

template <typename S>
std::ostream &operator<<(std::ostream &oss,
                         const ExplicitDynamicsDataTpl<S> &self) {
  oss << "ExplicitDynamicsData { ";
  oss << fmt::format("ndx: {:d},  ", self.ndx1);
  oss << fmt::format("nu:  {:d}", self.nu);
  oss << " }";
  return oss;
}

} // namespace proxddp

#include "proxddp/core/explicit-dynamics.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/explicit-dynamics.txx"
#endif
