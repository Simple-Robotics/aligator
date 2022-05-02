#pragma once
///   Base definitions for explicit integrators.


#include "proxddp/modelling/dynamics/integrator-base.hpp"


namespace proxddp
{
  namespace dynamics
  {
    
    /// @brief  Explicit integrators \f$ x_{k+1} = \phi(x_k, u_k) \f$.
    template<typename _Scalar>
    struct ExplicitIntegratorTpl : ExplicitDynamicsModelTpl<_Scalar>, IntegratorBaseTpl<_Scalar>
    {
    public:
      using Scalar = _Scalar;
      using IntegratorBase = IntegratorBaseTpl<Scalar>;
      using ContDynamics = ODEBaseTpl<Scalar>;

      inline const ContDynamics& continuous() const
      {
        return static_cast<const ContDynamics&>(cont_dynamics_);
      }

      ExplicitIntegratorTpl(const ContDynamics& cont_dynamics)
        : IntegratorBase(cont_dynamics)
        , ExplicitDynamicsModelTpl<Scalar>(cont_dynamics.ndx(), cont_dynamics.nu())
        {}


      shared_ptr<DynamicsDataTpl<Scalar>> createData() const
      {
        return std::make_shared<ExplicitIntegratorDataTpl<Scalar>>(*this);
      }

    protected:
      using IntegratorBase::cont_dynamics_;
    };

    template<typename _Scalar>
    struct ExplicitIntegratorDataTpl : IntegratorBaseDataTpl<_Scalar>, ExplicitDynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      using IntegratorBaseData = IntegratorBaseDataTpl<Scalar>;
      using IntegratorBaseData::cont_dynamics;

      ExplicitIntegratorDataTpl(const ExplicitIntegratorTpl<Scalar> integrator)
        : IntegratorBaseData(integrator)
        , ExplicitDynamicsDataTpl<Scalar>(integrator.ndx(), integrator.nu())
        {}
    };
    
  } // namespace dynamics
} // namespace proxddp

