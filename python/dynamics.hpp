#pragma once

#include "proxddp/python/fwd.hpp"
#include "proxddp/core/dynamics/continuous-base.hpp"
#include "proxddp/core/dynamics/base-ode.hpp"


namespace proxddp
{
  namespace python
  {
    namespace internal
    {
      template<class T = dynamics::ContinuousDynamicsTpl<context::Scalar>>
      struct PyContinuousDynamics : T, bp::wrapper<T>
      {
        using bp::wrapper<T>::get_override;
        using Data = typename T::Data;
        PROXNLP_DYNAMIC_TYPEDEFS(context::Scalar);

        template<typename... Args>
        PyContinuousDynamics(PyObject*&, Args&&... args)
          : T(std::forward<Args>(args)...) {}

        void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, const ConstVectorRef& xdot, Data& data) const
        { get_override("evaluate")(x, u, xdot, data); }

        void computeJacobians(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              const ConstVectorRef& xdot,
                              Data& data) const
        { get_override("computeJacobians")(x, u, xdot, data); }

      };


      struct PyODEBase :
          dynamics::ODEBaseTpl<context::Scalar>,
          bp::wrapper<dynamics::ODEBaseTpl<context::Scalar>>
      {
        PROXNLP_DYNAMIC_TYPEDEFS(context::Scalar);
        using Base = dynamics::ODEBaseTpl<context::Scalar>;
        using Data = dynamics::ODEBaseTpl<context::Scalar>::Data;

        template<typename ...Args>
        PyODEBase(PyObject*&, Args&&... args)
          : Base(std::forward<Args>(args)...) {}

        void forward(const ConstVectorRef& x,
                     const ConstVectorRef& u,
                     VectorRef xdot_out) const override
        {
          get_override("forward")(x, u, xdot_out);
        }

        void dForward(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      MatrixRef Jxdot_x,
                      MatrixRef Jxdot_u) const override
        {
          get_override("dForward")(x, u, Jxdot_x, Jxdot_u);
        }
      };

    } // namespace internal
    
  } // namespace python
} // namespace proxddp


