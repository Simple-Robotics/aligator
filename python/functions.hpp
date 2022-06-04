#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/core/function.hpp"
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
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

        // Use perfect forwarding to the FunctionBase constructors.
        template<typename... Args>
        PyStageFunction(PyObject*&, Args&&... args)
          : FunctionBase(std::forward<Args>(args)...) {}

        virtual void evaluate(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              const ConstVectorRef& y,
                              Data& data) const override
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, y, data); }

        virtual void computeJacobians(const ConstVectorRef& x,
                                      const ConstVectorRef& u,
                                      const ConstVectorRef& y,
                                      Data& data) const override
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeJacobians", x, u, y, data); }
        
        virtual void computeVectorHessianProducts(const ConstVectorRef& x,
                                                  const ConstVectorRef& u,
                                                  const ConstVectorRef& y,
                                                  const ConstVectorRef& lbda,
                                                  Data& data) const override
        { PROXDDP_PYTHON_OVERRIDE(void, FunctionBase, computeVectorHessianProducts, x, u, y, lbda, data); }

        shared_ptr<Data> createData() const override
        { PROXDDP_PYTHON_OVERRIDE(shared_ptr<Data>, FunctionBase, createData,); }

      };

      struct PyExplicitDynamicsModel :
        ExplicitDynamicsModelTpl<context::Scalar>,
        bp::wrapper<ExplicitDynamicsModelTpl<context::Scalar>>
      {
        using Scalar = context::Scalar;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

        template<typename... Args>
        PyExplicitDynamicsModel(Args&&... args)
          : ExplicitDynamicsModelTpl<Scalar>(std::forward<Args>(args)...) {}

        virtual void forward(const ConstVectorRef& x,
                             const ConstVectorRef& u,
                             VectorRef out) const override
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "forward", x, u, out) }

        virtual void dForward(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              MatrixRef Jx,
                              MatrixRef Ju) const override
        { PROXDDP_PYTHON_OVERRIDE_PURE(void, "dForward", x, u, Jx, Ju) }

      };
      
    } // namespace internal
    
  } // namespace python
} // namespace proxddp

