/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/multibody/fwd.hpp"
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace python {
void exposeFreeFwdDynamics() {
  using namespace aligator::dynamics;
  using context::Scalar;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using ContinuousDynamicsAbstract = ContinuousDynamicsAbstractTpl<Scalar>;
  using MultibodyFreeFwdData = MultibodyFreeFwdDataTpl<Scalar>;
  using MultibodyFreeFwdDynamics = MultibodyFreeFwdDynamicsTpl<Scalar>;
  using context::MultibodyPhaseSpace;

  PolymorphicMultiBaseVisitor<ODEAbstract, ContinuousDynamicsAbstract>
      ode_visitor;

  bp::class_<MultibodyFreeFwdDynamics, bp::bases<ODEAbstract>>(
      "MultibodyFreeFwdDynamics",
      "Free-space forward dynamics on multibodies using Pinocchio's ABA "
      "algorithm.",
      bp::init<MultibodyPhaseSpace, const context::MatrixXs &>(
          "Constructor where the actuation matrix is provided.",
          ("self"_a, "space", "actuation_matrix")))
      .def(bp::init<MultibodyPhaseSpace>(
          "Constructor without actuation matrix (assumed to be the (nu,nu) "
          "identity matrix).",
          ("self"_a, "space")))
      .add_property("ntau", &MultibodyFreeFwdDynamics::ntau,
                    "Torque dimension.")
      .add_property(
          "isUnderactuated", &MultibodyFreeFwdDynamics::isUnderactuated,
          "Whether the system is underactuated, i.e. if the actuation matrix "
          "rank is lower than the acceleration vector's dimension.")
      .add_property("actuationMatrixRank",
                    &MultibodyFreeFwdDynamics::getActuationMatrixRank,
                    "Get the rank of the actuation matrix.")
      .def(ode_visitor);

  bp::register_ptr_to_python<shared_ptr<MultibodyFreeFwdData>>();

  bp::class_<MultibodyFreeFwdData, bp::bases<ODEData>>("MultibodyFreeFwdData",
                                                       bp::no_init)
      .def_readwrite("tau", &MultibodyFreeFwdData::tau_)
      .def_readwrite("dtau_dx", &MultibodyFreeFwdData::dtau_dx_)
      .def_readwrite("dtau_du", &MultibodyFreeFwdData::dtau_du_)
      .def_readwrite("pin_data", &MultibodyFreeFwdData::pin_data_);
}
} // namespace python
} // namespace aligator
#endif
