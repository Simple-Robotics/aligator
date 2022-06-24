/// @file integrator-base.hpp
/// @brief Base definitions for numerical integrators.
#pragma once

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
     */ 
    template<typename _Scalar>
    struct IntegratorAbstractTpl : virtual DynamicsModelTpl<_Scalar>
    {
    public:
      using Scalar = _Scalar;
      using Base = DynamicsModelTpl<Scalar>;
      using BaseData = DynamicsDataTpl<Scalar>;
      using ContinuousDynamics = ContinuousDynamicsAbstractTpl<Scalar>;

      /// The underlying continuous dynamics.
      shared_ptr<ContinuousDynamics> cont_dynamics_;

      /// @brief  Return a reference to the underlying continuous dynamics.
      virtual inline const ContinuousDynamics& getContinuousDynamics() const { return *cont_dynamics_; }

      /// Constructor from instances of DynamicsType.
      explicit IntegratorAbstractTpl(const shared_ptr<ContinuousDynamics>& cont_dynamics)
        : Base(cont_dynamics->ndx(), cont_dynamics->nu())
        , cont_dynamics_(cont_dynamics) {}

      shared_ptr<BaseData> createData() const
      {
        return std::make_shared<IntegratorDataTpl<Scalar>>(*this);
      }
    };


    /// @brief  Data class for numerical integrators (IntegratorAbstractTpl).
    template<typename _Scalar>
    struct IntegratorDataTpl : virtual DynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      shared_ptr<ContinuousDynamicsDataTpl<Scalar>> continuous_data;

      explicit IntegratorDataTpl(const IntegratorAbstractTpl<Scalar>& integrator)
        : DynamicsDataTpl<Scalar>(integrator.ndx1, integrator.nu, integrator.ndx2, integrator.ndx2)
        , continuous_data(std::move(integrator.cont_dynamics_->createData()))
        {}

    };

  } // namespace dynamics
} // namespace proxddp

