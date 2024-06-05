/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include "aligator/python/fwd.hpp"
#include "aligator/python/functions.hpp"
#include "aligator/python/polymorphic-convertible.hpp"

#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/integrator-abstract.hpp"
#include "aligator/modelling/dynamics/integrator-midpoint.hpp"

namespace aligator {
namespace python {
using context::Scalar;
using namespace aligator::dynamics;
using context::DynamicsModel;
using context::IntegratorAbstract;

void exposeIntegrators() {
  using DAEType = ContinuousDynamicsAbstractTpl<Scalar>;

  using PyIntegratorAbstract = internal::PyStageFunction<IntegratorAbstract>;
  register_polymorphic_to_python<xyz::polymorphic<IntegratorAbstract>>();

  PolymorphicMultiBaseVisitor<DynamicsModel, IntegratorAbstract>
      conversions_visitor;

  bp::class_<PyIntegratorAbstract, bp::bases<DynamicsModel>,
             boost::noncopyable>("IntegratorAbstract",
                                 "Base class for numerical integrators.",
                                 bp::init<const xyz::polymorphic<DAEType> &>(
                                     "Construct the integrator from a DAE.",
                                     bp::args("self", "cont_dynamics")))
      .def_readwrite("continuous_dynamics",
                     &IntegratorAbstract::continuous_dynamics_,
                     "The underlying ODE or DAE.")
      .add_property("space",
                    bp::make_function(&IntegratorAbstract::space,
                                      bp::return_internal_reference<>()),
                    "Return the state manifold.")
      // allow casting custom Python class to polymorphic<base_classes>...
      .def(conversions_visitor);

  bp::register_ptr_to_python<shared_ptr<IntegratorDataTpl<Scalar>>>();
  bp::class_<IntegratorDataTpl<Scalar>, bp::bases<DynamicsDataTpl<Scalar>>>(
      "IntegratorData", "Base class for integrators' data.", bp::no_init)
      .def_readwrite("continuous_data",
                     &IntegratorDataTpl<Scalar>::continuous_data);

  using MidpointIntegrator = IntegratorMidpointTpl<Scalar>;
  bp::class_<MidpointIntegrator, bp::bases<IntegratorAbstract>>(
      "IntegratorMidpoint", bp::init<xyz::polymorphic<DAEType>, Scalar>(
                                bp::args("self", "dae", "timestep")))
      .def_readwrite("timestep", &MidpointIntegrator::timestep_, "Time step.")
      .def(conversions_visitor);

  bp::register_ptr_to_python<shared_ptr<IntegratorMidpointDataTpl<Scalar>>>();
  bp::class_<IntegratorMidpointDataTpl<Scalar>,
             bp::bases<IntegratorDataTpl<Scalar>>>("IntegratorMidpointData",
                                                   bp::no_init);
}

} // namespace python
} // namespace aligator
