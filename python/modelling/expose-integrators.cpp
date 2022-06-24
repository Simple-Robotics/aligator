
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

      using IntegratorAbstract = IntegratorAbstractTpl<Scalar>;
      using DAEType = ContinuousDynamicsAbstractTpl<Scalar>;

      /*
      using PyIntegratorAbstract = internal::PyStageFunction<IntegratorAbstract>;
      bp::class_<PyIntegratorAbstract, bp::bases<context::DynamicsModel>>(
        "IntegratorAbstract", "Base class for numerical integrators.",
        bp::init<const shared_ptr<DAEType>&>("Construct the integrator from a DAE.", bp::args("self", "cont_dynamics"))
      )
        .def("getContinuousDynamics", &IntegratorAbstract::getContinuousDynamics,
             "Get the underlying continuous dynamics.",
             bp::return_internal_reference<>());

      bp::class_<IntegratorEuler<Scalar>, bp::bases<IntegratorAbstract>>(
        "IntegratorEuler",
        "The explicit Euler integrator :math:`x' = x \\oplus \\Delta t f(x, u)`; "
        "this integrator has error :math:`O(\\Delta t)` "
        "in the time step :math:`\\Delta t`."
      )
        .def_readwrite("timestep", &IntegratorEuler<Scalar>::timestep_, "Time step.");
      */
    }
    
  } // namespace python
} // namespace proxddp


