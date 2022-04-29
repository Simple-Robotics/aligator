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
        // using Data = typename FunctionBase::Data;
        PROXNLP_FUNCTION_TYPEDEFS(Scalar)

        // Use perfect forwarding to the FunctionBase constructors.
        template<typename... Args>
        PyStageFunction(PyObject*&, Args&&... args)
          : FunctionBase(std::forward<Args>(args)...) {}

        void evaluate(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      const ConstVectorRef& y,
                      Data& data) const
        { get_override("evaluate")(x, u, y, data); }

        void computeJacobians(const ConstVectorRef& x,
                              const ConstVectorRef& u,
                              const ConstVectorRef& y,
                              Data& data) const
        { get_override("computeJacobians")(x, u, y, data); }
        
        void computeVectorHessianProducts(const ConstVectorRef& x,
                                          const ConstVectorRef& u,
                                          const ConstVectorRef& y,
                                          const ConstVectorRef& lbda,
                                          Data& data) const
        {
          if (bp::override f = get_override("computeVectorHessianProducts"))
          {
            f(x, u, y, lbda, data);
          } else {
            FunctionBase::computeVectorHessianProducts(x, u, y, lbda, data);
          }
        }

        shared_ptr<Data> createData() const
        {
          if (bp::override f = get_override("createData"))
          {
            return f();
          } else {
            return FunctionBase::createData();
          }
        }

      };

      struct PyExplicitDynamicalModel :
        ExplicitDynamicsModelTpl<context::Scalar>,
        bp::wrapper<ExplicitDynamicsModelTpl<context::Scalar>>
      {
        using Scalar = context::Scalar;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

        void forward(const ConstVectorRef& x,
                     const ConstVectorRef& u,
                     VectorRef out) const
        { get_override("forward")(x, u, out); }

        void dForward(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      MatrixRef Jx,
                      MatrixRef Ju) const
        { get_override("dForward")(x, u, Jx, Ju); }

      };
      
    } // namespace internal
    

    void exposeFunctions()
    {
      using context::Scalar;
      using context::DynamicsModel;
      using context::StageFunction;
      using internal::PyStageFunction;

      bp::class_<StageFunction, PyStageFunction<>, boost::noncopyable>(
        "StageFunction",
        "Base class for ternary functions f(x,u,x') on a stage of the problem.",
        bp::init<const int, const int, const int, const int>
                 (bp::args("self", "ndx1", "nu", "ndx2", "nr"))
      )
        .def(bp::init<const int, const int, const int>(bp::args("self", "ndx", "nu", "nr")))
        .def("evaluate",
             bp::pure_virtual(&StageFunction::evaluate),
             bp::args("self", "x", "u", "y", "data"))
        .def("computeJacobians",
             bp::pure_virtual(&StageFunction::computeJacobians),
             bp::args("self", "x", "u", "y", "data"))
        .def("computeVectorHessianProducts",
             &StageFunction::computeVectorHessianProducts,
             bp::args("self", "x", "u", "y", "lbda", "data"))
        .def("createData", &StageFunction::createData, "Create a data object.")
      ;

      bp::class_<context::FunctionData, shared_ptr<context::FunctionData>>(
        "FunctionData", "Data struct for holding data about functions.",
        bp::init<const int, const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2", "nr")
        )
      )
        .def_readonly("value", &context::FunctionData::value_,
                      "Function value.")
        .def_readonly("Jx", &context::FunctionData::Jx_,
                      "Jacobian with respect to $x$.")
        .def_readonly("Ju", &context::FunctionData::Ju_,
                      "Jacobian with respect to $u$.")
        .def_readonly("Jy", &context::FunctionData::Jy_,
                      "Jacobian with respect to $y$.")
        .def_readonly("Hxx", &context::FunctionData::Hxx_,
                      "Hessian with respect to $(x, x)$.")
        .def_readonly("Hxu", &context::FunctionData::Hxu_,
                      "Hessian with respect to $(x, u)$.")
        .def_readonly("Hxy", &context::FunctionData::Hxy_,
                      "Hessian with respect to $(x, y)$.")
        .def_readonly("Huu", &context::FunctionData::Huu_,
                      "Hessian with respect to $(u, u)$.")
        .def_readonly("Huy", &context::FunctionData::Huy_,
                      "Hessian with respect to $(x, y)$.")
        .def_readonly("Hyy", &context::FunctionData::Hyy_,
                      "Hessian with respect to $(y, y)$.")
      ;

      /** DYNAMICS **/

      bp::class_<DynamicsModel,
                 bp::bases<StageFunction>,
                 PyStageFunction<DynamicsModel>,
                 boost::noncopyable>(
        "DynamicsModel",
        "Dynamics models are specific ternary functions f(x,u,x') which map "
        "to the tangent bundle of the next state variable x'.",
        bp::init<const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2")
          )
      )
        .def(bp::init<const int, const int>(
          bp::args("self", "ndx", "nu")
          ))
      ;

      // bp::class_<internal::PyExplicitDynamicalModel,
      //            bp::bases<internal::PyDynWrap>, boost::noncopyable>
      // ("ExplicitDynamicsModel", "Explicit dynamics.", bp::no_init);

    }

  } // namespace python
} // namespace proxddp

