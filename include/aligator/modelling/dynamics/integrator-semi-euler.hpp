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
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;
  using Base::space_next_;

  /// Integration time step \f$h\f$.
  Scalar timestep_;

  IntegratorSemiImplEulerTpl(const shared_ptr<ODEType> &cont_dynamics,
                             const Scalar timestep);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const override;

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override;

  shared_ptr<StageFunctionDataTpl<Scalar>> createData() const override {
    return std::make_shared<Data>(this);
  }

  shared_ptr<StageFunctionDataTpl<Scalar>>
  createData(const CommonModelDataContainer &container) const override {
    return std::make_shared<Data>(this, container);
  }
};

template <typename Scalar>
struct IntegratorSemiImplDataTpl : ExplicitIntegratorDataTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ExplicitIntegratorDataTpl<Scalar>;
  using ODEData = ODEDataTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  MatrixXs Jtmp_xnext2;
  MatrixXs Jtmp_u;

  explicit IntegratorSemiImplDataTpl(
      const IntegratorSemiImplEulerTpl<Scalar> *integrator);
  IntegratorSemiImplDataTpl(
      const IntegratorSemiImplEulerTpl<Scalar> *integrator,
      const CommonModelDataContainer &container);

  using Base::dx_;
  using Base::Jtmp_xnext;
  using Base::Ju_;
  using Base::Jx_;
  using Base::xnext_;
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/integrator-semi-euler.hxx"
