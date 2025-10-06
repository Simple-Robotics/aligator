#pragma once
/// @file integrator-semi-impl-euler.hpp
/// @author Quentin Le Lidec
/// @brief Define the semi-implicit Euler integrator.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include "aligator/modelling/dynamics/integrator-explicit.hpp"

namespace aligator {
namespace dynamics {
template <typename Scalar> struct IntegratorSemiImplDataTpl;
/**
 *  @brief Semi-implicit Euler integrator \f$ x_{k+1} = x_k \oplus h f(x_k,
 * u_k)\f$.
 */
template <typename _Scalar>
struct IntegratorSemiImplEulerTpl : ExplicitIntegratorAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorAbstractTpl<Scalar>;
  using BaseData = ExplicitDynamicsDataTpl<Scalar>;
  using Data = IntegratorSemiImplDataTpl<Scalar>;
  using ODEType = ODEAbstractTpl<Scalar>;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  using Base::space_next_;

  /// Integration time step \f$h\f$.
  Scalar timestep_;

  IntegratorSemiImplEulerTpl(const xyz::polymorphic<ODEType> &cont_dynamics,
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
struct IntegratorSemiImplDataTpl : ExplicitIntegratorDataTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Model = IntegratorSemiImplEulerTpl<Scalar>;
  using Base = ExplicitIntegratorDataTpl<Scalar>;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;

  MatrixXs Jtmp_xnext2;
  MatrixXs Jtmp_u;

  explicit IntegratorSemiImplDataTpl(const Model &integrator)
      : Base(integrator)
      , Jtmp_xnext2(integrator.ndx1(), integrator.ndx1())
      , Jtmp_u(integrator.ndx1(), integrator.nu) {
    Jtmp_xnext2.setZero();
    Jtmp_u.setZero();
  }

  using Base::dx_;
  using Base::xnext_;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct IntegratorSemiImplEulerTpl<context::Scalar>;
extern template struct IntegratorSemiImplDataTpl<context::Scalar>;
#endif

} // namespace dynamics
} // namespace aligator
