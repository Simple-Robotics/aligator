#pragma once

#include "proxddp/modelling/dynamics/fwd.hpp"
#include "proxddp/modelling/dynamics/continuous-base.hpp"

#include "proxddp/modelling/dynamics/base-ode.hpp"
#include "proxddp/core/explicit-dynamics.hpp"


namespace proxddp
{
  namespace dynamics
  {
    
    /// @brief  Base class for (implicit) numerical integrators.
    ///
    /// Numerical integrators are instances DynamicsModelTpl
    /// which call into a ContinuousDynamicsTpl and construct an integration rule.
    template<typename _Scalar>
    struct IntegratorBaseTpl : DynamicsModelTpl<_Scalar>
    {
    public:
      using Scalar = _Scalar;
      using ContDynamics = ContinuousDynamicsTpl<Scalar>;

      /// @brief    Return a reference to the underlying continuous dynamics.
      virtual inline const ContDynamics& continuous() const { return cont_dynamics_; }

      /// Constructor from instances of DynamicsType.
      explicit IntegratorBaseTpl(const ContDynamics& cont_dynamics)
        : DynamicsModelTpl<Scalar>(cont_dynamics.ndx(), cont_dynamics.nu())
        , cont_dynamics_(cont_dynamics) {}

      shared_ptr<DynamicsDataTpl<Scalar>> createData() const
      {
        return std::make_shared<IntegratorBaseDataTpl<Scalar>>(*this);
      }

    protected:
      /// The underlying continuous dynamics.
      const ContDynamics& cont_dynamics_;
    };


    /// @brief  Data class for numerical integrators IntegratorBaseTpl.
    template<typename _Scalar>
    struct IntegratorBaseDataTpl : DynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      shared_ptr<ContinuousDynamicsDataTpl<Scalar>> cont_data;

      IntegratorBaseDataTpl(const IntegratorBaseTpl<Scalar>& integrator)
        : DynamicsDataTpl<Scalar>(integrator.ndx1, integrator.nu, integrator.ndx2, integrator.ndx2)
        , cont_data(std::move(integrator.continuous().createData()))
        {}

    };


    //// EXPLICIT INTEGRATORS


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
    struct ExplicitIntegratorDataTpl : ExplicitDynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      shared_ptr<ContinuousDynamicsDataTpl<Scalar>> cont_data;

      ExplicitIntegratorDataTpl(const ExplicitIntegratorTpl<Scalar> integrator)
        : ExplicitDynamicsDataTpl<Scalar>(integrator.ndx(), integrator.nu())
        , cont_data(std::move(integrator.continuous().createData()))
        {}

    };

  } // namespace dynamics
} // namespace proxddp

