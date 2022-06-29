#pragma once
/// @file integrator-abstract.hpp
/// @brief Base definitions for numerical integrators.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxddp/modelling/dynamics/continuous-base.hpp"


namespace proxddp
{
  namespace dynamics
  {
    
    /**
     * @brief  Base class for (implicit) numerical integrators.
     *
     * @details Numerical integrators are instances DynamicsModelTpl which call into a
     *          ContinuousDynamicsAbstractTpl and construct an integration rule.
     *          Their aim is to provide a discretization for DAEs
     *          \f[
     *            f(x(t), u(t), \dot{x}(t)) = 0
     *          \f]
     *          as
     *          \f[
     *            \Phi(x_k, u_k, x_{k+1}) = 0.
     *          \f]
     */
    template<typename _Scalar>
    struct IntegratorAbstractTpl : DynamicsModelTpl<_Scalar>
    {
    public:
      using Scalar = _Scalar;
      using Base = DynamicsModelTpl<Scalar>;
      using BaseData = DynamicsDataTpl<Scalar>;
      using ContinuousDynamics = ContinuousDynamicsAbstractTpl<Scalar>;

      /// The underlying continuous dynamics.
      shared_ptr<ContinuousDynamics> continuous_dynamics_;

      /// Constructor from instances of DynamicsType.
      explicit IntegratorAbstractTpl(const shared_ptr<ContinuousDynamics>& cont_dynamics);
      virtual ~IntegratorAbstractTpl() = default;
      shared_ptr<BaseData> createData() const;
    };


    /// @brief  Data class for numerical integrators (IntegratorAbstractTpl).
    template<typename _Scalar>
    struct IntegratorDataTpl : DynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      shared_ptr<ContinuousDynamicsDataTpl<Scalar>> continuous_data;

      explicit IntegratorDataTpl(const IntegratorAbstractTpl<Scalar>* integrator);
      virtual ~IntegratorDataTpl() = default;
    };

  } // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/integrator-abstract.hxx"
