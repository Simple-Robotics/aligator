#pragma once
/// @file base-ode.hpp
/// @brief Defines a class representing ODEs.

#include "proxddp/core/dynamics/continuous-base.hpp"


namespace proxddp
{
  namespace dynamics
  {
    /** @brief   Base class for ODE dynamics \f$ \dot{x} = f(x, u) \f$.
     * @details  Formulated as a DAE (for ContinuousDynamicsTpl), this class models
     *           \f[
     *              f(x, u) - \dot{x}.
     *           \f]
     */            
    template<typename _Scalar>
    struct ODEBaseTpl : ContinuousDynamicsTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

      using BaseType = ContinuousDynamicsTpl<_Scalar>;
      using Data = typename BaseType::Data;
      using BaseType::BaseType; // use parent ctor

      using SpecificData = ODEDataTpl<Scalar>;

      virtual void forward(const ConstVectorRef& x,
                           const ConstVectorRef& u,
                           VectorRef xdot_out) const = 0;

      virtual void dForward(const ConstVectorRef& x,
                            const ConstVectorRef& u,
                            MatrixRef Jxdot_x,
                            MatrixRef Jxdot_u) const = 0;

      /** Declare overrides **/

      void evaluate(const ConstVectorRef& x,
                    const ConstVectorRef& u,
                    const ConstVectorRef& xdot,
                    Data& data) const override;

      void computeJacobians(const ConstVectorRef& x,
                            const ConstVectorRef& u,
                            const ConstVectorRef& xdot,
                            Data& data) const override;

      shared_ptr<Data> createData() const override
      {
        return std::make_shared<SpecificData>(this->ndx(), this->nu());
      }

    };

    template<typename _Scalar>
    struct ODEDataTpl : ContinuousDynamicsDataTpl<_Scalar>
    {
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
      using Base = ContinuousDynamicsDataTpl<Scalar>;

      /// Value of the derivative \f$\dot{x}\f$
      VectorXs xdot_;

      ODEDataTpl(const int ndx, const int nu)
        : Base(ndx, nu)
        , xdot_(ndx)
      {
        xdot_.setZero();
        this->Jxdot_ = -MatrixXs::Identity(ndx, ndx);
      }
    };
    
  } // namespace dynamics
} // namespace proxddp

#include "proxddp/core/dynamics/base-ode.hxx"

