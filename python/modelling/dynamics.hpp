#pragma once

#include "proxddp/python/functions.hpp"
#include "proxddp/core/explicit-dynamics.hpp"


namespace proxddp
{
  namespace python
  {
    namespace internal
    {

      template<class ExplicitBase = ExplicitDynamicsModelTpl<context::Scalar>>
      struct PyExplicitDynamics : ExplicitBase, bp::wrapper<ExplicitBase>
      {
        using Scalar = context::Scalar;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
        using BaseData = ExplicitDynamicsDataTpl<Scalar>;

        template<typename... Args>
        PyExplicitDynamics(Args&&... args) : ExplicitBase(args...) {}

        virtual void forward(const ConstVectorRef& x,
                             const ConstVectorRef& u,
                             BaseData& data) const
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "forward", x, u, data); }

        virtual void dForward(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              BaseData& data) const
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, data); }

      };

    } // namespace internal
  } // namespace python
} // namespace proxddp