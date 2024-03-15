/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#ifdef ALIGATOR_PINOCCHIO_V3

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"

namespace aligator {
namespace python {
namespace {

using RigidConstraintModel =
    pinocchio::RigidConstraintModelTpl<context::Scalar, 0>;
using RigidConstraintData =
    pinocchio::RigidConstraintDataTpl<context::Scalar, 0>;

using RigidConstraintModelVector =
    PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);
using RigidConstraintDataVector =
    PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);

} // namespace

void exposeConstrainedFwdDynamics() {
  using namespace aligator::dynamics;
  using context::Scalar;
  using ODEData = ODEDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using MultibodyConstraintFwdData = MultibodyConstraintFwdDataTpl<Scalar>;
  using MultibodyConstraintFwdDynamics =
      MultibodyConstraintFwdDynamicsTpl<Scalar>;

  bp::class_<MultibodyConstraintFwdDynamics, bp::bases<ODEAbstract>>(
      "MultibodyConstraintFwdDynamics",
      "Constraint forward dynamics using Pinocchio.",
      bp::init<const shared_ptr<proxsuite::nlp::MultibodyPhaseSpace<Scalar>> &,
               const context::MatrixXs &, const RigidConstraintModelVector &,
               const pinocchio::ProximalSettingsTpl<Scalar> &>(
          bp::args("self", "space", "actuation_matrix", "constraint_models",
                   "prox_settings")))
      .def_readwrite("constraint_models",
                     &MultibodyConstraintFwdDynamics::constraint_models_)
      .add_property("ntau", &MultibodyConstraintFwdDynamics::ntau,
                    "Torque dimension.");

  bp::register_ptr_to_python<shared_ptr<MultibodyConstraintFwdData>>();

  bp::class_<MultibodyConstraintFwdData, bp::bases<ODEData>>(
      "MultibodyConstraintFwdData", bp::no_init)
      .add_property(
          "multibody_data",
          bp::make_getter(&MultibodyConstraintFwdData::multibody_data_,
                          bp::return_internal_reference<>()));
}
} // namespace python
} // namespace aligator

#endif
