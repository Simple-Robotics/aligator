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
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  /// The underlying continuous dynamics.
  shared_ptr<ContinuousDynamics> continuous_dynamics_;

  /// Constructor from instances of DynamicsType.
  explicit IntegratorAbstractTpl(
      const shared_ptr<ContinuousDynamics> &cont_dynamics);

  void configure(
      CommonModelBuilderContainer &common_buider_container) const override {
    continuous_dynamics_->configure(common_buider_container);
  }

  shared_ptr<BaseData> createData() const override;
  shared_ptr<BaseData>
  createData(const CommonModelDataContainer &container) const override;
};

/// @brief  Data class for numerical integrators (IntegratorAbstractTpl).
template <typename _Scalar>
struct IntegratorDataTpl : DynamicsDataTpl<_Scalar> {
  using Scalar = _Scalar;
  using Base = DynamicsDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;
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

  explicit IntegratorDataTpl(const IntegratorAbstractTpl<Scalar> *integrator);
  explicit IntegratorDataTpl(const IntegratorAbstractTpl<Scalar> *integrator,
                             const CommonModelDataContainer &container);
  virtual ~IntegratorDataTpl() = default;
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/integrator-abstract.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/integrator-abstract.txx"
#endif
