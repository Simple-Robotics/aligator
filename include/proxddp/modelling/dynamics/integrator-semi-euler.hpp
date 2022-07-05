#pragma once
/// @file integrator-semi-impl-euler.hpp
/// @author Quentin Le Lidec
/// @brief Define the semi-implicit Euler integrator.

#include "proxddp/modelling/dynamics/integrator-explicit.hpp"

namespace proxddp {
namespace dynamics {
/**
 *  @brief Semi-implicit Euler integrator \f$ x_{k+1} = x_k \oplus h f(x_k,
 * u_k)\f$.
 */
template <typename _Scalar>
struct IntegratorSemiImplEulerTpl : ExplicitIntegratorAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorAbstractTpl<Scalar>;
  using Data = IntegratorSemiImplDataTpl<Scalar>;
  using ODEType = ODEAbstractTpl<Scalar>;

  /// Integration time step \f$h\f$.
  Scalar timestep_;

  IntegratorSemiImplEulerTpl(const shared_ptr<ODEType> &cont_dynamics,
                             const Scalar timestep);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ExplicitDynamicsDataTpl<Scalar> &data) const;

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ExplicitDynamicsDataTpl<Scalar> &data) const;
};

template <typename Scalar>
struct IntegratorSemiImplDataTpl : ExplicitIntegratorDataTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorDataTpl<Scalar>;
  using ODEData = ODEDataTpl<Scalar>;
  shared_ptr<ODEData> continuous_data2;

  MatrixXs Jtmp_xnext2;
  MatrixXs Jtmp_u;

  explicit IntegratorSemiImplDataTpl(
      const IntegratorSemiImplEulerTpl<Scalar> *integrator);

  using Base::dx_;
  using Base::Jtmp_xnext;
  using Base::Ju_;
  using Base::Jx_;
  using Base::xnext_;
};

} // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/integrator-semi-euler.hxx"