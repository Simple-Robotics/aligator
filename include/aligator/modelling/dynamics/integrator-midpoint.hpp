/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-abstract.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar> struct IntegratorMidpointDataTpl;

/**
 * @brief Midpoint integration rule.
 *
 * @details The rule is described, for general DAEs, as
 *          \f[
 *             \phi(x_k, u_k, x_{k+1}) =
 *             g(\hat{x}_0, u_k, \frac{x_{k+1}\ominus x_k}{h}) = 0.
 *          \f]
 *          where \f$\hat{x}_0 = \mathrm{Interp}_{1/2}(x_k, x_{k+1})\f$.
 *          Even for ODEs, it is still an implicit integration rule.
 *
 *          The Jacobians are:
 *          \f[
 *            \frac{\partial f}{\partial z} = \frac{\partial g}{\partial
 * \hat{x}_0} \frac{\partial \hat{x}_0}{\partial z} + \frac{\partial g}{\partial
 * z} + \frac{\partial g}{\partial v} \frac{\partial v}{\partial z} \f]
 */
template <typename _Scalar>
struct IntegratorMidpointTpl : IntegratorAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = IntegratorAbstractTpl<Scalar>;
  using ContinuousDynamics = ContinuousDynamicsAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Data = IntegratorMidpointDataTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  Scalar timestep_;

  IntegratorMidpointTpl(const shared_ptr<ContinuousDynamics> &cont_dynamics,
                        const Scalar timestep);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y,
                DynamicsDataTpl<Scalar> &data) const override;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y,
                        DynamicsDataTpl<Scalar> &data) const override;

  shared_ptr<DynamicsDataTpl<Scalar>> createData() const override;
  shared_ptr<DynamicsDataTpl<Scalar>>
  createData(const CommonModelDataContainer &container) const override;
};

template <typename _Scalar>
struct IntegratorMidpointDataTpl : IntegratorDataTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = IntegratorDataTpl<Scalar>;
  using CommonModelContainer = CommonModelContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;

  VectorXs x1_;
  VectorXs dx1_;
  MatrixXs J_v_0;
  MatrixXs J_v_1;
  MatrixXs Jtm0;
  MatrixXs Jtm1;

  CommonModelContainer common_models;
  CommonModelDataContainer common_datas;

  explicit IntegratorMidpointDataTpl(
      const IntegratorMidpointTpl<Scalar> *integrator)
      : Base(integrator), x1_(integrator->space().neutral()), dx1_(this->ndx1),
        J_v_0(this->ndx1, this->ndx1), J_v_1(this->ndx1, this->ndx1),
        Jtm0(this->ndx1, this->ndx1), Jtm1(this->ndx1, this->ndx1) {
    x1_.setZero();
    dx1_.setZero();
    J_v_0.setZero();
    J_v_1.setZero();
    Jtm0.setZero();
    Jtm1.setZero();

    CommonModelBuilderContainer model_builders;
    integrator->continuous_dynamics_->configure(model_builders);
    common_models = model_builders.createCommonModelContainer();
    common_datas = common_models.createData();
    this->continuous_data =
        integrator->continuous_dynamics_->createData(common_datas);
  }

  IntegratorMidpointDataTpl(const IntegratorMidpointTpl<Scalar> *integrator,
                            const CommonModelDataContainer &container)
      : Base(integrator, container), x1_(integrator->space().neutral()),
        dx1_(this->ndx1), J_v_0(this->ndx1, this->ndx1),
        J_v_1(this->ndx1, this->ndx1), Jtm0(this->ndx1, this->ndx1),
        Jtm1(this->ndx1, this->ndx1) {
    x1_.setZero();
    dx1_.setZero();
    J_v_0.setZero();
    J_v_1.setZero();
    Jtm0.setZero();
    Jtm1.setZero();

    CommonModelBuilderContainer model_builders;
    integrator->continuous_dynamics_->configure(model_builders);
    common_models = model_builders.createCommonModelContainer();
    common_datas = common_models.createData();
    this->continuous_data =
        integrator->continuous_dynamics_->createData(common_datas);
  }
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/integrator-midpoint.hxx"
