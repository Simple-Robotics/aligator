#pragma once
/// @file integrator-explicit.hpp
/// @brief  Base definitions for explicit integrators.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA


#include "proxddp/modelling/dynamics/integrator-abstract.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/modelling/dynamics/ode-abstract.hpp"


namespace proxddp
{
  namespace dynamics
  {
    
    /**
     * @brief   Explicit integrators \f$x_{k+1} = f(x_k, u_k) \f$.
     * @details This class of integrator mostly applies to integrating ODE models \f$\dot{x} = \phi(x,u)\f$.
     */ 
    template<typename _Scalar>
    struct ExplicitIntegratorAbstractTpl : ExplicitDynamicsModelTpl<_Scalar>, IntegratorAbstractTpl<_Scalar>
    {
      using Scalar = _Scalar;
      using IntegratorAbstract = IntegratorAbstractTpl<Scalar>;
      using ODEType = ODEAbstractTpl<Scalar>;
      using BaseExplicit = ExplicitDynamicsModelTpl<Scalar>;

      using BaseExplicit::evaluate;
      using BaseExplicit::computeJacobians;
      using BaseExplicit::ndx1;
      using BaseExplicit::ndx2;
      using BaseExplicit::nu;

      shared_ptr<ODEType> cont_dynamics_;

      explicit ExplicitIntegratorAbstractTpl(const shared_ptr<ODEType>& cont_dynamics)
        : IntegratorAbstract(cont_dynamics)
        , BaseExplicit(cont_dynamics.ndx(), cont_dynamics.nu())
        {}


      shared_ptr<DynamicsDataTpl<Scalar>> createData() const
      {
        return std::make_shared<ExplicitIntegratorDataTpl<Scalar>>(*this);
      }

    };

    template<typename _Scalar>
    struct ExplicitIntegratorDataTpl : IntegratorDataTpl<_Scalar>, ExplicitDynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      using IntegratorData = IntegratorDataTpl<Scalar>;
      using ExplicitData = ExplicitDynamicsDataTpl<Scalar>;
      using ExplicitData::Jx_;
      using ExplicitData::Ju_;
      using ExplicitData::Jy_;

      ExplicitIntegratorDataTpl(const ExplicitIntegratorAbstractTpl<Scalar>& integrator)
        : IntegratorData(integrator)
        , ExplicitData(integrator.ndx1, integrator.nu)
        {}
    };
    
  } // namespace dynamics
} // namespace proxddp

