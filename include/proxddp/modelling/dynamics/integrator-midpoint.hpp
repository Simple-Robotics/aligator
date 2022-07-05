#pragma once

#include "proxddp/modelling/dynamics/integrator-abstract.hpp"

#include <stdexcept>

namespace proxddp {
namespace dynamics {
template <typename Scalar> struct IntegratorIntegratorMidpointDataTpl;

/**
 * @brief Midpoint integration rule.
 *
 * @todo finish implementing computeJacobians()
 *
 * @details The rule is described, for general DAEs, as
 *          \f[
 *             g(\hat{x}_0, u_k, \frac{x_{k+1}\ominus x_k}{h}) = 0.
 *          \f]
 *          where \f$\hat{x}_0 = \mathrm{Interp}_{1/2}(x_k, x_{k+1})\f$.
 *          Even for ODEs, it is still an implicit integration rule.
 */
template <typename _Scalar>
struct IntegratorMidpointTpl : IntegratorAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = IntegratorAbstractTpl<Scalar>;
  using ContinuousDynamics = ContinuousDynamicsAbstractTpl<Scalar>;

  Scalar timestep_;

  explicit IntegratorMidpointTpl(
      const shared_ptr<ContinuousDynamics> &cont_dynamics,
      const Scalar timestep) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, DynamicsDataTpl<Scalar> &data) const {
    IntegratorIntegratorMidpointDataTpl<Scalar> &d =
        static_cast<IntegratorIntegratorMidpointDataTpl<Scalar> &>(data);
    const ContinuousDynamics *contdyn = this->continuous_dynamics_.get();
    const ManifoldAbstractTpl<Scalar> &space = contdyn->space();
    ContinuousDynamicsDataTpl<Scalar> *contdata = d.continuous_data.get();
    // define x1 = midpoint of x,y
    space.difference(x, y, d.xdot_);
    d.xdot_ /= timestep_;
    // define xdot = (y-x) / timestep
    space.interpolate(x, y, 0.5, d.x1_);
    // evaluate on (x1, u, xdot)
    contdyn->evaluate(d.x1_, u, d.xdot_, *contdata);
    d.value_ = contdata->value_;
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y,
                        DynamicsDataTpl<Scalar> &data) const {
    IntegratorIntegratorMidpointDataTpl<Scalar> &d =
        static_cast<IntegratorIntegratorMidpointDataTpl<Scalar> &>(data);
    const ContinuousDynamics *contdyn = this->continuous_dynamics_.get();
    const ManifoldAbstractTpl<Scalar> &space = contdyn->space();
    ContinuousDynamicsDataTpl<Scalar> *contdata = d.continuous_data.get();

    // d.x1_ contains midpoint of x,y
    // compute jacobians
    contdyn->computeJacobians(d.x1_, u, d.xdot_, *contdata);
    throw std::exception("Implementation not finished.");
  }
};

template <typename _Scalar>
struct IntegratorIntegratorMidpointDataTpl : IntegratorDataTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = IntegratorDataTpl<Scalar>;

  VectorXs x1_;
  VectorXs dx1_;

  explicit IntegratorIntegratorMidpointDataTpl(
      const IntegratorMidpointTpl<Scalar> *integrator)
      : Base(integrator), x1_(integrator->ndx) {}
};

} // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/integrator-midpoint.hxx"
