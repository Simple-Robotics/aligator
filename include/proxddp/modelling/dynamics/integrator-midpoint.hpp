#pragma once

#include "proxddp/modelling/dynamics/integrator-abstract.hpp"

namespace proxddp {
namespace dynamics {
template <typename Scalar> struct IntegratorMidpointDataTpl;

/**
 * @brief Midpoint integration rule.
 *
 * @todo finish implementing computeJacobians()
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = IntegratorAbstractTpl<Scalar>;
  using ContinuousDynamics = ContinuousDynamicsAbstractTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Data = IntegratorMidpointDataTpl<Scalar>;

  Scalar timestep_;

  IntegratorMidpointTpl(const shared_ptr<ContinuousDynamics> &cont_dynamics,
                        const Scalar timestep);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, DynamicsDataTpl<Scalar> &data) const {
    IntegratorMidpointDataTpl<Scalar> &d =
        static_cast<IntegratorMidpointDataTpl<Scalar> &>(data);
    const ContinuousDynamics *contdyn = this->continuous_dynamics_.get();
    const Manifold &space = contdyn->space();
    ContinuousDynamicsDataTpl<Scalar> *contdata = d.continuous_data.get();
    // define xdot = (y-x) / timestep
    space.difference(x, y, d.xdot_);
    d.xdot_ /= timestep_;
    // define x1 = midpoint of x,y
    space.interpolate(x, y, 0.5, d.x1_);
    space.difference(x, d.x1_, d.dx1_);
    // evaluate on (x1, u, xdot)
    contdyn->evaluate(d.x1_, u, d.xdot_, *contdata);
    d.value_ = contdata->value_;
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y,
                        DynamicsDataTpl<Scalar> &data) const {
    IntegratorMidpointDataTpl<Scalar> &d =
        static_cast<IntegratorMidpointDataTpl<Scalar> &>(data);
    const ContinuousDynamics *contdyn = this->continuous_dynamics_.get();
    const Manifold &space = contdyn->space();
    ContinuousDynamicsDataTpl<Scalar> *contdata = d.continuous_data.get();

    auto dx = d.dx1_ * 2;
    // jacobians of xdot estimate
    space.Jdifference(x, y, d.J_v_0, 0);
    space.Jdifference(x, y, d.J_v_1, 1);
    d.J_v_0 /= timestep_;
    d.J_v_1 /= timestep_;
    // d.x1_ contains midpoint of x,y
    // compute jacobians
    contdyn->computeJacobians(d.x1_, u, d.xdot_, *contdata);

    // bring the Jacobian in arg1 from xmid to x
    space.JintegrateTransport(x, d.dx1_, contdata->Jx_, 1);
    data.Jx_ = 0.5 * contdata->Jx_ + contdata->Jxdot_ * d.J_v_0;

    data.Ju_ = contdata->Ju_;

    // bring the Jacobian in x = y - dx to y
    space.JintegrateTransport(y, -dx, contdata->Jx_, 1);
    data.Jy_ = 0.5 * contdata->Jx_ + contdata->Jxdot_ * d.J_v_1;
  }

  shared_ptr<DynamicsDataTpl<Scalar>> createData() const;
};

template <typename _Scalar>
struct IntegratorMidpointDataTpl : IntegratorDataTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = IntegratorDataTpl<Scalar>;

  VectorXs x1_;
  VectorXs dx1_;
  MatrixXs J_v_0;
  MatrixXs J_v_1;

  explicit IntegratorMidpointDataTpl(
      const IntegratorMidpointTpl<Scalar> *integrator)
      : Base(integrator), x1_(integrator->space().neutral()), dx1_(this->ndx1),
        J_v_0(this->ndx1, this->ndx1), J_v_1(this->ndx1, this->ndx1) {
    x1_.setZero();
    J_v_0.setZero();
    J_v_1.setZero();
  }
};

} // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/integrator-midpoint.hxx"
