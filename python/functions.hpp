#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/core/node-function.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"


namespace proxddp
{
  namespace python
  {
    namespace internal
    {
      /// Wrapper from StageFunction objects and their children that does
      /// not require the child wrappers to create more virtual function overrides.
      ///
      /// Using a templating technique from Pybind11's docs:
      /// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance
      template<class FunctionBase = context::StageFunction>
      struct PyStageFunction : FunctionBase, bp::wrapper<FunctionBase>
      {
        using bp::wrapper<FunctionBase>::get_override;
        using Scalar = typename FunctionBase::Scalar;
        using Data = FunctionDataTpl<Scalar>;
        PROXNLP_FUNCTION_TYPEDEFS(Scalar)

        // Use perfect forwarding to the FunctionBase constructors.
        template<typename... Args>
        PyStageFunction(PyObject*&, Args&&... args)
          : FunctionBase(std::forward<Args>(args)...) {}

        void evaluate(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      const ConstVectorRef& y,
                      Data& data) const override
        { get_override("evaluate")(x, u, y, data); }

        void computeJacobians(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              const ConstVectorRef& y,
                              Data& data) const override
        { get_override("computeJacobians")(x, u, y, data); }
        
        void computeVectorHessianProducts(const ConstVectorRef& x,
                                          const ConstVectorRef& u,
                                          const ConstVectorRef& y,
                                          const ConstVectorRef& lbda,
                                          Data& data) const override
        {
          if (bp::override f = get_override("computeVectorHessianProducts"))
          {
            f(x, u, y, lbda, data);
          } else {
            FunctionBase::computeVectorHessianProducts(x, u, y, lbda, data);
          }
        }

        shared_ptr<Data> createData() const override
        {
          if (bp::override f = get_override("createData"))
          {
            return f();
          } else {
            return FunctionBase::createData();
          }
        }

      };

      struct PyExplicitDynamicsModel :
        ExplicitDynamicsModelTpl<context::Scalar>,
        bp::wrapper<ExplicitDynamicsModelTpl<context::Scalar>>
      {
        using Scalar = context::Scalar;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

        template<typename... Args>
        PyExplicitDynamicsModel(Args&&... args)
          : ExplicitDynamicsModelTpl<Scalar>(std::forward<Args>(args)...) {}

        void forward(const ConstVectorRef& x,
                     const ConstVectorRef& u,
                     VectorRef out) const override
        { get_override("forward")(x, u, out); }

        void dForward(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      MatrixRef Jx,
                      MatrixRef Ju) const override
        { get_override("dForward")(x, u, Jx, Ju); }

      };
      
    } // namespace internal
    
  } // namespace python
} // namespace proxddp

