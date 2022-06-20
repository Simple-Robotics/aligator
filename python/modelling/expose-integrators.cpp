
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/functions.hpp"

#include "proxddp/modelling/dynamics/integrator-base.hpp"
#include "proxddp/modelling/dynamics/integrator-euler.hpp"


namespace proxddp
{
  namespace python
  {
    
    void exposeIntegrators()
    {
      using context::Scalar;
      using namespace proxddp::dynamics;

      using IntegratorBase = IntegratorBaseTpl<Scalar>;

      using PyIntegratorBase = internal::PyStageFunction<IntegratorBase>;
      bp::class_<PyIntegratorBase,
                 bp::bases<context::DynamicsModel>
                 >
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


