
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/functions.hpp"

#include "proxddp/modelling/dynamics/continuous-base.hpp"
#include "proxddp/modelling/dynamics/base-ode.hpp"
#include "proxddp/modelling/dynamics/integrator-base.hpp"
#include "proxddp/modelling/dynamics/euler.hpp"


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
        using Scalar = typename T::Scalar;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

        template<typename... Args>
        PyContinuousDynamics(PyObject*&, Args&&... args)
          : T(std::forward<Args>(args)...) {}

        void forward(const ConstVectorRef& x, const ConstVectorRef& u, const ConstVectorRef& xdot, VectorRef out) const
        {
          get_override("forward")(x, u, xdot, out);
        }

        void dForward(const ConstVectorRef& x,
                      const ConstVectorRef& u,
                      const ConstVectorRef& xdot,
                      MatrixRef Jx,
                      MatrixRef Ju) const
        {
          get_override("dForward")(x, u, xdot, Jx, Ju);
        }

      };
      
    } // namespace internal
    
    void exposeIntegrators()
    {
      using context::Scalar;
      using dynamics::ContinuousDynamicsTpl;
      using dynamics::ODEBaseTpl;
      using dynamics::IntegratorBaseTpl;
      using dynamics::IntegratorEuler;

      using IntegratorBase = IntegratorBaseTpl<Scalar>;
      using ContinuousBase = ContinuousDynamicsTpl<Scalar>;

      bp::class_<
        ContinuousBase,
        internal::PyContinuousDynamics<>
        >
      (
        "ContinuousDynamics", "Base class for continuous dynamics/DAE models.",
        bp::init<const context::Manifold&, const int>(
          bp::args("self", "space", "nu")
          )
      )
        .def("forward", &ContinuousBase::forward)
        .def("dForward", &ContinuousBase::dForward)
        .def("create_data", &ContinuousBase::createData, "Instantiate a data holder.")
      ;

      {
        using ContData = dynamics::ContinuousDynamicsDataTpl<Scalar>;
        bp::register_ptr_to_python<shared_ptr<ContData>>();
        bp::class_<ContData>
        (
          "ContinuousDynamicsData", bp::no_init
        )
          .def_readonly("value", &ContData::value_)
          .def_readonly("Jx", &ContData::Jx_)
          .def_readonly("Ju", &ContData::Ju_)
          .def_readonly("Jxdot", &ContData::Jxdot_)
          ;
      }

      // bp::class_<ODEBaseTpl<Scalar>, bp::bases<ContinuousBase>>(
      //   "ODEDynamics", "Continuous dynamics described by ordinary differential equations (ODEs).",
      //   bp::no_init
      // );

      bp::class_<IntegratorBase,
                 bp::bases<context::DynamicsModel>,
                 internal::PyStageFunction<IntegratorBase>>
      (
        "IntegratorBase", "Base class for numerical integrators.",
        bp::init<const IntegratorBase::ContDynamics&>(
          bp::args("self", "cont_dynamics")
          )
      )
        .def("continuous", &IntegratorBase::continuous,
             "Get the underlying continuous dynamics.",
             bp::return_internal_reference<>())
      ;

      // bp::class_<IntegratorEuler<Scalar>, bp::bases<IntegratorBase>>(
      //   "Euler", "Explicit Euler integrator."
      // );

    }
    
  } // namespace python
} // namespace proxddp


