/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-explicit.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar> struct IntegratorRK2DataTpl;

/** @brief  Second-order Runge-Kutta integrator.
 *
 * \f{eqnarray*}{
 *    x_{k+1} = x_k \oplus h f(x^{(1)}, u_k),\\
 *    x^{(1)} = x_k \oplus \frac h2 f(x_k, u_k)
 * \f}
 *
 */
template <typename _Scalar>
struct IntegratorRK2Tpl : ExplicitIntegratorAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorAbstractTpl<Scalar>;
  using BaseData = ExplicitDynamicsDataTpl<Scalar>;
  using Data = IntegratorRK2DataTpl<Scalar>;
  using ODEType = typename Base::ODEType;

  Scalar timestep_;

  IntegratorRK2Tpl(const xyz::polymorphic<ODEType> &cont_dynamics,
                   const Scalar timestep);
  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }
};

template <typename Scalar>
struct IntegratorRK2DataTpl : ExplicitIntegratorDataTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorDataTpl<Scalar>;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  shared_ptr<ODEData> continuous_data2;

  VectorXs x1_;
  VectorXs dx1_;

  explicit IntegratorRK2DataTpl(const IntegratorRK2Tpl<Scalar> &integrator)
      : Base(integrator)
      , x1_(integrator.space_next().neutral())
      , dx1_(this->ndx1) {
    continuous_data2 = integrator.ode_->createData();
    dx1_.setZero();
  }

  using Base::dx_;
  using Base::Jtmp_xnext;
  using Base::Ju;
  using Base::Jx;
  using Base::xnext_;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct IntegratorRK2Tpl<context::Scalar>;
extern template struct IntegratorRK2DataTpl<context::Scalar>;
#endif
} // namespace dynamics
} // namespace aligator
