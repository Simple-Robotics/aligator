/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#ifdef ALIGATOR_PINOCCHIO_V3

#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include "aligator/python/polymorphic-convertible.hpp"

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
  using context::ContinuousDynamicsAbstract;
  using context::ODEAbstract;
  using context::ODEData;
  using context::Scalar;
  using MultibodyConstraintFwdData = MultibodyConstraintFwdDataTpl<Scalar>;
  using MultibodyConstraintFwdDynamics =
      MultibodyConstraintFwdDynamicsTpl<Scalar>;

  convertibleToPolymorphicBases<MultibodyConstraintFwdDynamics, ODEAbstract,
                                ContinuousDynamicsAbstract>();
  bp::class_<MultibodyConstraintFwdDynamics, bp::bases<ODEAbstract>>(
      "MultibodyConstraintFwdDynamics",
      "Constraint forward dynamics using Pinocchio.",
      bp::init<const proxsuite::nlp::MultibodyPhaseSpace<Scalar> &,
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
      .def_readwrite("tau", &MultibodyConstraintFwdData::tau_)
      .def_readwrite("dtau_dx", &MultibodyConstraintFwdData::dtau_dx_)
      .def_readwrite("dtau_du", &MultibodyConstraintFwdData::dtau_du_)
      .def_readwrite("pin_data", &MultibodyConstraintFwdData::pin_data_)
      .def_readwrite("constraint_datas",
                     &MultibodyConstraintFwdData::constraint_datas_);
}
} // namespace python
} // namespace aligator

#endif
