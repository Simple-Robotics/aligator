
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/functions.hpp"
#include "proxddp/python/modelling/dynamics.hpp"

#include "proxddp/modelling/dynamics/integrator-abstract.hpp"
#include "proxddp/modelling/dynamics/integrator-euler.hpp"
#include "proxddp/modelling/dynamics/integrator-rk2.hpp"
#include "proxddp/modelling/dynamics/integrator-semi-impl-euler.hpp"

namespace proxddp {
namespace python {

void exposeIntegrators() {
  using context::Scalar;
  using namespace proxddp::dynamics;

  using IntegratorAbstract = IntegratorAbstractTpl<Scalar>;
  using DAEType = ContinuousDynamicsAbstractTpl<Scalar>;
  using ODEType = ODEAbstractTpl<Scalar>;

  //// GENERIC INTEGRATORS

  using PyIntegratorAbstract = internal::PyStageFunction<IntegratorAbstract>;
  bp::class_<PyIntegratorAbstract, bp::bases<context::DynamicsModel>>(
      "IntegratorAbstract", "Base class for numerical integrators.",
      bp::init<const shared_ptr<DAEType> &>(
          "Construct the integrator from a DAE.",
          bp::args("self", "cont_dynamics")))
      .def_readwrite("differential_dynamics",
                     &IntegratorAbstract::continuous_dynamics_,
                     "The underlying ODE or DAE.");

  bp::register_ptr_to_python<shared_ptr<IntegratorDataTpl<Scalar>>>();
  bp::class_<IntegratorDataTpl<Scalar>, bp::bases<DynamicsDataTpl<Scalar>>>(
      "IntegratorData", "Base class for integrators' data.", bp::no_init)
      .def_readwrite("continuous_data",
                     &IntegratorDataTpl<Scalar>::continuous_data);

  //// EXPLICIT INTEGRATORS

  using ExplicitIntegratorAbstract = ExplicitIntegratorAbstractTpl<Scalar>;
  using PyExplicitIntegrator =
      internal::PyExplicitDynamics<ExplicitIntegratorAbstract>;

  bp::class_<PyExplicitIntegrator, bp::bases<ExplicitDynamicsModelTpl<Scalar>>>(
      "ExplicitIntegratorAbstract", bp::no_init)
      .def_readwrite("differential_dynamics", &ExplicitIntegratorAbstract::ode_,
                     "The underlying differential equation.");

  bp::register_ptr_to_python<shared_ptr<ExplicitIntegratorDataTpl<Scalar>>>();
  bp::class_<ExplicitIntegratorDataTpl<Scalar>,
             bp::bases<ExplicitDynamicsDataTpl<Scalar>>>(
      "ExplicitIntegratorData", "Data struct for explicit time integrators.",
      bp::no_init)
      .def_readwrite("continuous_data",
                     &ExplicitIntegratorDataTpl<Scalar>::continuous_data);

  bp::class_<IntegratorEulerTpl<Scalar>, bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorEuler",
      "The explicit Euler integrator :math:`x' = x \\oplus \\Delta t f(x, u)`; "
      "this integrator has error :math:`O(\\Delta t)` "
      "in the time step :math:`\\Delta t`.",
      bp::init<shared_ptr<ODEType>, Scalar>(
          bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorEulerTpl<Scalar>::timestep_,
                     "Time step.");

  bp::class_<IntegratorSemiImplEulerTpl<Scalar>,
             bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorSemiImplEuler", "The semi implicit Euler integrator.",
      bp::init<shared_ptr<ODEType>, Scalar>(
          bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorSemiImplEulerTpl<Scalar>::timestep_,
                     "Time step.");

  bp::class_<IntegratorRK2Tpl<Scalar>, bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorRK2", bp::init<shared_ptr<ODEType>, Scalar>(
                           bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorRK2Tpl<Scalar>::timestep_,
                     "Time step.");
}

} // namespace python
} // namespace proxddp
