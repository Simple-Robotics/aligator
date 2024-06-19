/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#ifdef ALIGATOR_PINOCCHIO_V3

#include "aligator/modelling/multibody/contact-force.hpp"

namespace aligator {
namespace python {

namespace {

using RigidConstraintModel =
    pinocchio::RigidConstraintModelTpl<context::Scalar, 0>;
using RigidConstraintModelVector =
    PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);

void exposeContactForceResidual() {
  using namespace aligator;
  using context::Scalar;

  using StageFunction = StageFunctionTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using ContactForceResidual = ContactForceResidualTpl<Scalar>;
  using ContactForceData = ContactForceDataTpl<Scalar>;

  bp::class_<ContactReference>(
      "ContactReference",
      "Contains contact reference in the contact model vector",
      bp::init<const RigidConstraintModelVector &, std::size_t>(
          bp::args("self", "contact_models", "contact_index")))
      .def_readonly("contact_index", &ContactReference::contact_index)
      .def_readonly("force_index", &ContactReference::force_index)
      .def_readonly("force_size", &ContactReference::force_size);

  bp::class_<ContactForceResidual, bp::bases<StageFunction>>(
      "ContactForceResidual", "Compute contact force error.",
      bp::init<const int, const int, const ContactReference &>(
          bp::args("self", "ndx", "nu", "contact_ref")))
      .def("getReference", &ContactForceResidual::getReference,
           bp::args("self"), bp::return_internal_reference<>())
      .def("setReference", &ContactForceResidual::setReference,
           bp::args("self", "ref"))
      .def("getContactReference", &ContactForceResidual::getContactReference,
           bp::args("self"), bp::return_internal_reference<>());

  bp::class_<ContactForceData, bp::bases<StageFunctionData>>(
      "ContactForceData", "ContactForceResidual Data", bp::no_init)
      .add_property("multibody_data",
                    bp::make_getter(&ContactForceData::multibody_data_,
                                    bp::return_internal_reference<>()));
}

} // namespace

void exposeContactForce() { exposeContactForceResidual(); }

} // namespace python
} // namespace aligator

#endif // ALIGATOR_PINOCCHIO_V3
