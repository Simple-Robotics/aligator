/// @file continuous-dynamics-abstract.hpp
/// @brief Base definitions for continuous dynamics.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/modelling/dynamics/fwd.hpp"
#include "aligator/core/manifold-base.hpp"
#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>

namespace aligator {
namespace dynamics {

///  @brief Continuous dynamics described by differential-algebraic equations
/// (DAEs) \f$F(\dot{x}, x, u) = 0\f$.
///
/// @details Continuous dynamics described as \f$ f(x, u, \dot{x}) = 0 \f$.
///          The codimension of this function is the same as that of \f$x\f$.
template <typename _Scalar> struct ContinuousDynamicsAbstractTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using ManifoldPtr = xyz::polymorphic<Manifold>;
  using Data = ContinuousDynamicsDataTpl<Scalar>;

  /// State space.
  ManifoldPtr space_;
  /// Control space dimension.
  const int nu_;

  inline int ndx() const { return space_->ndx(); }
  inline int nu() const { return nu_; }

  /// @brief  Return a reference to the state space.
  inline const Manifold &space() const { return *space_; }

  ContinuousDynamicsAbstractTpl(ManifoldPtr space, const int nu);

  virtual ~ContinuousDynamicsAbstractTpl() = default;

  /// @brief   Evaluate the vector field at a point \f$(x, u)\f$.
  /// @param   x The input state variable.
  /// @param   u The input control variable.
  /// @param   xdot Derivative of the state.
  /// @param[out] data The output data object.
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xdot, Data &data) const = 0;

  /// @brief  Differentiate the vector field.
  /// @param   x The input state variable.
  /// @param   u The input control variable.
  /// @param   xdot Derivative of the state.
  /// @param[out] data The output data object.
  virtual void computeJacobians(const ConstVectorRef &x,
                                const ConstVectorRef &u,
                                const ConstVectorRef &xdot,
                                Data &data) const = 0;

  /// @brief  Create a data holder instance.
  virtual shared_ptr<Data> createData() const;
};

/// @brief  Data struct for ContinuousDynamicsAbstractTpl.
template <typename _Scalar> struct ContinuousDynamicsDataTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  /// Residual value \f$e = f(x,u,\dot{x})\f$
  VectorXs value_;
  /// Derivative \f$\partial f/{\partial x}\f$
  MatrixXs Jx_;
  /// Derivative \f$\partial f/{\partial u}\f$
  MatrixXs Ju_;
  /// Derivative \f$\partial f/\partial\dot{x}\f$
  MatrixXs Jxdot_;
  /// Time derivative \f$\dot{x} = f(x, u)\f$, output of ODE model
  VectorXs xdot_;

  ContinuousDynamicsDataTpl(const int ndx, const int nu);

  // marks this type as polymorphic; required for Boost.Python
  virtual ~ContinuousDynamicsDataTpl() = default;
};

} // namespace dynamics
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/continuous-dynamics-abstract.txx"
#endif
