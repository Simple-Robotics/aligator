#pragma once
/// @file integrator-abstract.hpp
/// @brief Base definitions for numerical integrators.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include "aligator/modelling/dynamics/continuous-dynamics-abstract.hpp"
#include "aligator/core/dynamics.hpp"

namespace aligator {
namespace dynamics {

/**
 * @brief  Base class for (implicit) numerical integrators.
 *
 * @details Numerical integrators are instances DynamicsModelTpl which call into
 * a ContinuousDynamicsAbstractTpl and construct an integration rule. Their aim
 * is to provide a discretization for DAEs \f[ f(x(t), u(t), \dot{x}(t)) = 0 \f]
 *          as
 *          \f[
 *            \Phi(x_k, u_k, x_{k+1}) = 0.
 *          \f]
 */
template <typename _Scalar>
struct IntegratorAbstractTpl : DynamicsModelTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  using Base = DynamicsModelTpl<Scalar>;
  using BaseData = DynamicsDataTpl<Scalar>;
  using ContinuousDynamics = ContinuousDynamicsAbstractTpl<Scalar>;

  /// The underlying continuous dynamics.
  xyz::polymorphic<ContinuousDynamics> continuous_dynamics_;

  template <typename U> U *getDynamics() {
    return dynamic_cast<U *>(&*continuous_dynamics_);
  }
  template <typename U> const U *getDynamics() const {
    return dynamic_cast<const U *>(&*continuous_dynamics_);
  }

  /// Constructor from instances of DynamicsType.
  explicit IntegratorAbstractTpl(
      const xyz::polymorphic<ContinuousDynamics> &cont_dynamics);
  virtual ~IntegratorAbstractTpl() = default;
  shared_ptr<BaseData> createData() const;
};

/// @brief  Data class for numerical integrators (IntegratorAbstractTpl).
template <typename _Scalar>
struct IntegratorDataTpl : DynamicsDataTpl<_Scalar> {
  using Scalar = _Scalar;
  using Base = DynamicsDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  shared_ptr<ContinuousDynamicsDataTpl<Scalar>> continuous_data;

  using Base::Huu_;
  using Base::Huy_;
  using Base::Hxu_;
  using Base::Hxx_;
  using Base::Hxy_;
  using Base::Hyy_;
  using Base::Ju_;
  using Base::Jx_;
  using Base::Jy_;
  using Base::value_;

  /// Value of the time-derivative to use in the integration rule.
  VectorXs xdot_;

  explicit IntegratorDataTpl(const IntegratorAbstractTpl<Scalar> &integrator);
  virtual ~IntegratorDataTpl() = default;
};

} // namespace dynamics
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/integrator-abstract.txx"
#endif
