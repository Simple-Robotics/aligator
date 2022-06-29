
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/functions.hpp"
#include "proxddp/python/modelling/dynamics.hpp"

#include "proxddp/modelling/dynamics/integrator-abstract.hpp"
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
      using ODEType = ODEAbstractTpl<Scalar>;

      //// GENERIC INTEGRATORS

      using PyIntegratorAbstract = internal::PyStageFunction<IntegratorAbstract>;
      bp::class_<PyIntegratorAbstract, bp::bases<context::DynamicsModel>>(
        "IntegratorAbstract", "Base class for numerical integrators.",
        bp::init<const shared_ptr<DAEType>&>("Construct the integrator from a DAE.", bp::args("self", "cont_dynamics")))
        .def_readwrite("differential_dynamics", &IntegratorAbstract::continuous_dynamics_, "The underlying ODE or DAE.");


      //// EXPLICIT INTEGRATORS

      using ExplicitIntegratorAbstract = ExplicitIntegratorAbstractTpl<Scalar>;
      using PyExplicitIntegrator = internal::PyExplicitDynamics<ExplicitIntegratorAbstract>;


      bp::class_<PyExplicitIntegrator, bp::bases<ExplicitDynamicsModelTpl<Scalar>>>(
        "ExplicitIntegratorAbstract", bp::no_init)
        .def_readwrite("differential_dynamics", &ExplicitIntegratorAbstract::ode_, "The underlying differential equation.");

      bp::class_<IntegratorEuler<Scalar>, bp::bases<ExplicitIntegratorAbstract>>(
        "IntegratorEuler",
        "The explicit Euler integrator :math:`x' = x \\oplus \\Delta t f(x, u)`; "
        "this integrator has error :math:`O(\\Delta t)` "
        "in the time step :math:`\\Delta t`.",
        bp::init<const shared_ptr<ODEType>&, Scalar>(bp::args("self", "ode", "timestep"))
      )
        .def_readwrite("timestep", &IntegratorEuler<Scalar>::timestep_, "Time step.");

    }
    
  } // namespace python
} // namespace proxddp


