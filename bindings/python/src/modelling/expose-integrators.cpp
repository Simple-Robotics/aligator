
#include "aligator/python/fwd.hpp"
#include "aligator/python/functions.hpp"
#include "aligator/python/modelling/explicit-dynamics.hpp"

#include "aligator/modelling/dynamics/integrator-abstract.hpp"
#include "aligator/modelling/dynamics/integrator-euler.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/dynamics/integrator-midpoint.hpp"

namespace aligator {
namespace python {

void exposeIntegrators() {
  using context::Scalar;
  using namespace aligator::dynamics;

  using IntegratorAbstract = IntegratorAbstractTpl<Scalar>;
  using DAEType = ContinuousDynamicsAbstractTpl<Scalar>;
  using ODEType = ODEAbstractTpl<Scalar>;

  //// GENERIC INTEGRATORS

  using PyIntegratorAbstract = internal::PyStageFunction<IntegratorAbstract>;
  bp::class_<PyIntegratorAbstract, bp::bases<context::DynamicsModel>,
             boost::noncopyable>("IntegratorAbstract",
                                 "Base class for numerical integrators.",
                                 bp::init<const shared_ptr<DAEType> &>(
                                     "Construct the integrator from a DAE.",
                                     bp::args("self", "cont_dynamics")))
      .def_readwrite("continuous_dynamics",
                     &IntegratorAbstract::continuous_dynamics_,
                     "The underlying ODE or DAE.")
      .add_property("space",
                    bp::make_function(&IntegratorAbstract::space,
                                      bp::return_internal_reference<>()),
                    "Return the state manifold.");

  bp::register_ptr_to_python<shared_ptr<IntegratorDataTpl<Scalar>>>();
  bp::class_<IntegratorDataTpl<Scalar>, bp::bases<DynamicsDataTpl<Scalar>>>(
      "IntegratorData", "Base class for integrators' data.", bp::no_init)
      .def_readwrite("continuous_data",
                     &IntegratorDataTpl<Scalar>::continuous_data);

  //// EXPLICIT INTEGRATORS

  using ExplicitIntegratorAbstract = ExplicitIntegratorAbstractTpl<Scalar>;
  using PyExplicitIntegrator =
      internal::PyExplicitDynamics<ExplicitIntegratorAbstract>;

  bp::class_<PyExplicitIntegrator, bp::bases<ExplicitDynamicsModelTpl<Scalar>>,
             boost::noncopyable>("ExplicitIntegratorAbstract",
                                 bp::init<const shared_ptr<ODEType> &>(
                                     "Construct the integrator from an ODE.",
                                     bp::args("self", "cont_dynamics")))
      .def_readonly("nx2", &ExplicitIntegratorAbstract::nx2,
                    "Next state dimension.")
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
  bp::class_<IntegratorSemiImplDataTpl<Scalar>,
             bp::bases<ExplicitIntegratorDataTpl<Scalar>>>(
      "IntegratorSemiImplData", bp::no_init);

  bp::class_<IntegratorRK2Tpl<Scalar>, bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorRK2", bp::init<shared_ptr<ODEType>, Scalar>(
                           bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorRK2Tpl<Scalar>::timestep_,
                     "Time step.");

  bp::class_<IntegratorRK2DataTpl<Scalar>,
             bp::bases<ExplicitIntegratorDataTpl<Scalar>>>("IntegratorRK2Data",
                                                           bp::no_init);

  using MidpointType = IntegratorMidpointTpl<Scalar>;
  bp::class_<MidpointType, bp::bases<IntegratorAbstract>>(
      "IntegratorMidpoint", bp::init<shared_ptr<DAEType>, Scalar>(
                                bp::args("self", "dae", "timestep")))
      .def_readwrite("timestep", &MidpointType::timestep_, "Time step.");

  bp::register_ptr_to_python<shared_ptr<IntegratorMidpointDataTpl<Scalar>>>();
  bp::class_<IntegratorMidpointDataTpl<Scalar>,
             bp::bases<IntegratorDataTpl<Scalar>>>("IntegratorMidpointData",
                                                   bp::no_init);
}

} // namespace python
} // namespace aligator
