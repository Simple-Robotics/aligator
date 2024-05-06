/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO

#include "aligator/python/fwd.hpp"
#include "aligator/python/modelling/multibody-utils.hpp"
#include "aligator/modelling/multibody/contact-force.hpp"

namespace aligator {
namespace python {

using context::ConstVectorRef;
using context::MultibodyPhaseSpace;
using context::PinModel;
using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using RigidConstraintModel =
    pinocchio::RigidConstraintModelTpl<context::Scalar, 0>;
using RigidConstraintData =
    pinocchio::RigidConstraintDataTpl<context::Scalar, 0>;

using RigidConstraintModelVector =
    PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);
using RigidConstraintDataVector =
    PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);

void exposeContactForce() {
  using ContactForceResidual = ContactForceResidualTpl<Scalar>;
  using ContactForceData = ContactForceDataTpl<Scalar>;

  bp::class_<ContactForceResidual, bp::bases<StageFunction>>(
      "ContactForceResidual",
      "A residual function :math:`r(x) = v_{j,xy} e^{-s z_j}` where :math:`j` "
      "is a given frame index.",
      bp::no_init)
      .def(bp::init<int, PinModel, const context::MatrixXs &,
                    const RigidConstraintModelVector &,
                    const pinocchio::ProximalSettingsTpl<Scalar> &,
                    const context::Vector6s &, int>(
          bp::args("self", "ndx", "model", "actuation_matrix",
                   "constraint_models", "prox_settings", "fref", "contact_id")))
      .def(FrameAPIVisitor<ContactForceResidual>())
      .def_readwrite("constraint_models",
                     &ContactForceResidual::constraint_models_);

  bp::class_<ContactForceData, bp::bases<StageFunctionData>>("ContactForceData",
                                                             bp::no_init)
      .def_readwrite("tau", &ContactForceData::tau_)
      .def_readwrite("pin_data", &ContactForceData::pin_data_)
      .def_readwrite("constraint_datas", &ContactForceData::constraint_datas_);
}

} // namespace python
} // namespace aligator

#endif
