/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include "aligator/python/modelling/explicit-dynamics.hpp"
#include "aligator/python/polymorphic-convertible.hpp"

#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/integrator-euler.hpp"
#include "aligator/modelling/dynamics/integrator-rk2.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"

namespace aligator {
namespace python {
using context::Scalar;
using namespace aligator::dynamics;
using context::DynamicsModel;
using context::ExplicitDynamics;
using context::ExplicitDynamicsData;
using context::ExplicitIntegratorAbstract;
using context::ExplicitIntegratorData;
using context::IntegratorAbstract;
using xyz::polymorphic;

void exposeExplicitIntegrators() {
  using ODEType = ODEAbstractTpl<Scalar>;

  //// EXPLICIT INTEGRATORS

  using PyExplicitIntegrator =
      internal::PyExplicitDynamics<ExplicitIntegratorAbstract>;

  PolymorphicMultiBaseVisitor<ExplicitIntegratorAbstract, ExplicitDynamics,
                              DynamicsModel>
      conversions_visitor;

  register_polymorphic_to_python<polymorphic<ExplicitIntegratorAbstract>>();
  bp::class_<PyExplicitIntegrator, bp::bases<ExplicitDynamicsModelTpl<Scalar>>,
             boost::noncopyable>("ExplicitIntegratorAbstract",
                                 bp::init<const polymorphic<ODEType> &>(
                                     "Construct the integrator from an ODE.",
                                     bp::args("self", "cont_dynamics")))
      .def_readonly("nx2", &ExplicitIntegratorAbstract::nx2,
                    "Next state dimension.")
      .def_readwrite("differential_dynamics", &ExplicitIntegratorAbstract::ode_,
                     "The underlying differential equation.")
      // required visitor to allow casting custom Python integrator to
      // polymorphic<Base>
      .def(conversions_visitor);

  bp::register_ptr_to_python<shared_ptr<ExplicitIntegratorData>>();
  bp::class_<ExplicitIntegratorData, bp::bases<ExplicitDynamicsData>>(
      "ExplicitIntegratorData", "Data struct for explicit time integrators.",
      bp::no_init)
      .def_readwrite("continuous_data",
                     &ExplicitIntegratorData::continuous_data);

  bp::class_<IntegratorEulerTpl<Scalar>, bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorEuler",
      "The explicit Euler integrator :math:`x' = x \\oplus \\Delta t f(x, u)`; "
      "this integrator has error :math:`O(\\Delta t)` "
      "in the time step :math:`\\Delta t`.",
      bp::init<xyz::polymorphic<ODEType>, Scalar>(
          bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorEulerTpl<Scalar>::timestep_,
                     "Time step.")
      .def(conversions_visitor);

  bp::class_<IntegratorSemiImplEulerTpl<Scalar>,
             bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorSemiImplEuler", "The semi implicit Euler integrator.",
      bp::init<xyz::polymorphic<ODEType>, Scalar>(
          bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorSemiImplEulerTpl<Scalar>::timestep_,
                     "Time step.")
      .def(conversions_visitor);
  bp::class_<IntegratorSemiImplDataTpl<Scalar>,
             bp::bases<ExplicitIntegratorData>>("IntegratorSemiImplData",
                                                bp::no_init);

  bp::class_<IntegratorRK2Tpl<Scalar>, bp::bases<ExplicitIntegratorAbstract>>(
      "IntegratorRK2", bp::init<xyz::polymorphic<ODEType>, Scalar>(
                           bp::args("self", "ode", "timestep")))
      .def_readwrite("timestep", &IntegratorRK2Tpl<Scalar>::timestep_,
                     "Time step.")
      .def(conversions_visitor);

  bp::class_<IntegratorRK2DataTpl<Scalar>, bp::bases<ExplicitIntegratorData>>(
      "IntegratorRK2Data", bp::no_init);
}

} // namespace python
} // namespace aligator
