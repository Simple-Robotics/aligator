#pragma once

#include "proxddp/modelling/dynamics/integrator-explicit.hpp"

namespace proxddp {
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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorAbstractTpl<Scalar>;
  using BaseData = ExplicitDynamicsDataTpl<Scalar>;
  using Data = IntegratorRK2DataTpl<Scalar>;
  using ODEType = typename Base::ODEType;
  using Base::space_next_;

  Scalar timestep_;

  IntegratorRK2Tpl(const shared_ptr<ODEType> &cont_dynamics,
                   const Scalar timestep);
  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<FunctionDataTpl<Scalar>> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  Scalar dt_2_ = 0.5 * timestep_;
};

template <typename Scalar>
struct IntegratorRK2DataTpl : ExplicitIntegratorDataTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorDataTpl<Scalar>;
  using ODEData = ODEDataTpl<Scalar>;
  shared_ptr<ODEData> continuous_data2;

  VectorXs x1_;
  VectorXs dx1_;

  explicit IntegratorRK2DataTpl(const IntegratorRK2Tpl<Scalar> *integrator);

  using Base::dx_;
  using Base::Jtmp_xnext;
  using Base::Ju_;
  using Base::Jx_;
  using Base::xnext_;
};

} // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/integrator-rk2.hxx"
