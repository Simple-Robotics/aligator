#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/modelling/dynamics/continuous-base.hpp"
#include "proxddp/modelling/dynamics/ode-abstract.hpp"


namespace proxddp
{
  namespace python
  {
    namespace internal
    {
      struct PyExplicitDynamicsModel :
        ExplicitDynamicsModelTpl<context::Scalar>,
        bp::wrapper<ExplicitDynamicsModelTpl<context::Scalar>>
      {
        using Scalar = context::Scalar;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
        using Data = ExplicitDynamicsDataTpl<Scalar>;

        template<typename... Args>
        PyExplicitDynamicsModel(Args&&... args)
          : ExplicitDynamicsModelTpl<Scalar>(std::forward<Args>(args)...) {}

        virtual void forward(const ConstVectorRef& x,
                             const ConstVectorRef& u,
                             Data& data) const override
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "forward", x, u, data) }

        virtual void dForward(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              Data& data) const override
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, data) }

      };
      
      template<class T = dynamics::ContinuousDynamicsAbstractTpl<context::Scalar>>
      struct PyContinuousDynamics : T, bp::wrapper<T>
      {
        using bp::wrapper<T>::get_override;
        using Data = typename T::Data;
        PROXNLP_DYNAMIC_TYPEDEFS(context::Scalar);

        template<typename... Args>
        PyContinuousDynamics(Args&&... args)
          : T(std::forward<Args>(args)...) {}

        void evaluate(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      const ConstVectorRef& xdot,
                      Data& data) const
        { get_override("evaluate")(x, u, xdot, data); }

        void computeJacobians(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              const ConstVectorRef& xdot,
                              Data& data) const
        { get_override("computeJacobians")(x, u, xdot, data); }

      };


      struct PyODEBase :
          dynamics::ODEAbstractTpl<context::Scalar>,
          bp::wrapper<dynamics::ODEAbstractTpl<context::Scalar>>
      {
        PROXNLP_DYNAMIC_TYPEDEFS(context::Scalar);
        using Base = dynamics::ODEAbstractTpl<context::Scalar>;
        using Data = dynamics::ODEAbstractTpl<context::Scalar>::Data;

        template<typename ...Args>
        PyODEBase(Args&&... args)
          : Base(std::forward<Args>(args)...) {}

        void forward(const ConstVectorRef& x,
                     const ConstVectorRef& u,
                     Data& data) const override
        {
          get_override("forward")(x, u, data);
        }

        void dForward(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      Data& data) const override
        {
          get_override("dForward")(x, u, data);
        }
      };

    } // namespace internal
    
  } // namespace python
} // namespace proxddp


