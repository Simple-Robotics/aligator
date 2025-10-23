/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO

#include "aligator/python/fwd.hpp"
#include "aligator/python/modelling/multibody-utils.hpp"

#include "aligator/modelling/multibody/contact-force.hpp"
#include "aligator/modelling/multibody/multibody-wrench-cone.hpp"
#include "aligator/modelling/multibody/multibody-friction-cone.hpp"

namespace aligator {
namespace python {

using Vector3or6 = Eigen::Matrix<double, -1, 1, Eigen::ColMajor, 6, 1>;
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

const PolymorphicMultiBaseVisitor<StageFunction> func_visitor;

void exposeContactForce() {
  using ContactForceResidual = ContactForceResidualTpl<Scalar>;
  using ContactForceData = ContactForceDataTpl<Scalar>;

  using MultibodyWrenchConeResidual = MultibodyWrenchConeResidualTpl<Scalar>;
  using MultibodyWrenchConeData = MultibodyWrenchConeDataTpl<Scalar>;

  using MultibodyFrictionConeResidual =
      MultibodyFrictionConeResidualTpl<Scalar>;
  using MultibodyFrictionConeData = MultibodyFrictionConeDataTpl<Scalar>;

  bp::class_<ContactForceResidual, bp::bases<StageFunction>>(
      "ContactForceResidual",
      "A residual function :math:`r(x) = v_{j,xy} e^{-s z_j}` where :math:`j` "
      "is a given frame index.",
      bp::no_init)
      .def(bp::init<int, PinModel, const context::MatrixXs &,
                    const RigidConstraintModelVector &,
                    const pinocchio::ProximalSettingsTpl<Scalar> &,
                    const Eigen::VectorXd &, std::string_view>(bp::args(
          "self", "ndx", "model", "actuation_matrix", "constraint_models",
          "prox_settings", "fref", "contact_name")))
      .def(func_visitor)
      .def("getReference", &ContactForceResidual::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target force.")
      .def("setReference", &ContactForceResidual::setReference,
           bp::args("self", "fnew"), "Set the target force.")
      .def_readwrite("constraint_models",
                     &ContactForceResidual::constraint_models_);

  bp::class_<ContactForceData, bp::bases<StageFunctionData>>("ContactForceData",
                                                             bp::no_init)
      .def_readwrite("tau", &ContactForceData::tau_)
      .def_readwrite("pin_data", &ContactForceData::pin_data_)
      .def_readwrite("constraint_datas", &ContactForceData::constraint_datas_);

  bp::class_<MultibodyWrenchConeResidual, bp::bases<StageFunction>>(
      "MultibodyWrenchConeResidual", "A residual function :math:`r(x) = Af` ",
      bp::no_init)
      .def(bp::init<int, PinModel, const context::MatrixXs &,
                    const RigidConstraintModelVector &,
                    const pinocchio::ProximalSettingsTpl<Scalar> &,
                    std::string_view, const double, const double, const double>(
          bp::args("self", "ndx", "model", "actuation_matrix",
                   "constraint_models", "prox_settings", "contact_name", "mu",
                   "half_length", "half_width")))
      .def(func_visitor)
      .def_readwrite("constraint_models",
                     &MultibodyWrenchConeResidual::constraint_models_);

  bp::class_<MultibodyWrenchConeData, bp::bases<StageFunctionData>>(
      "MultibodyWrenchConeData", bp::no_init)
      .def_readwrite("tau", &MultibodyWrenchConeData::tau_)
      .def_readwrite("pin_data", &MultibodyWrenchConeData::pin_data_)
      .def_readwrite("constraint_datas",
                     &MultibodyWrenchConeData::constraint_datas_);

  bp::class_<MultibodyFrictionConeResidual, bp::bases<StageFunction>>(
      "MultibodyFrictionConeResidual", "A residual function :math:`r(x) = Af` ",
      bp::no_init)
      .def(bp::init<int, PinModel, const context::MatrixXs &,
                    const RigidConstraintModelVector &,
                    const pinocchio::ProximalSettingsTpl<Scalar> &,
                    std::string_view, const double>(
          bp::args("self", "ndx", "model", "actuation_matrix",
                   "constraint_models", "prox_settings", "contact_name", "mu")))
      .def(func_visitor)
      .def_readwrite("constraint_models",
                     &MultibodyFrictionConeResidual::constraint_models_);

  bp::class_<MultibodyFrictionConeData, bp::bases<StageFunctionData>>(
      "MultibodyFrictionConeData", bp::no_init)
      .def_readwrite("tau", &MultibodyFrictionConeData::tau_)
      .def_readwrite("pin_data", &MultibodyFrictionConeData::pin_data_)
      .def_readwrite("constraint_datas",
                     &MultibodyFrictionConeData::constraint_datas_);
}

} // namespace python
} // namespace aligator

#endif
