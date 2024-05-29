/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace python {
void exposeFreeFwdDynamics() {
  using namespace aligator::dynamics;
  using context::Scalar;
  using ODEData = ODEDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using ContinuousDynamicsAbstract = ContinuousDynamicsAbstractTpl<Scalar>;
  using MultibodyFreeFwdData = MultibodyFreeFwdDataTpl<Scalar>;
  using MultibodyFreeFwdDynamics = MultibodyFreeFwdDynamicsTpl<Scalar>;
  using proxsuite::nlp::MultibodyPhaseSpace;

  using StateManifold = MultibodyPhaseSpace<Scalar>;

  bp::implicitly_convertible<MultibodyFreeFwdDynamics,
                             xyz::polymorphic<ContinuousDynamicsAbstract>>();
  bp::implicitly_convertible<MultibodyFreeFwdDynamics,
                             xyz::polymorphic<ODEAbstract>>();
  bp::class_<MultibodyFreeFwdDynamics, bp::bases<ODEAbstract>>(
      "MultibodyFreeFwdDynamics",
      "Free-space forward dynamics on multibodies using Pinocchio's ABA "
      "algorithm.",
      bp::init<StateManifold, const context::MatrixXs &>(
          "Constructor where the actuation matrix is provided.",
          bp::args("self", "space", "actuation_matrix")))
      .def(bp::init<StateManifold>(
          "Constructor without actuation matrix (assumed to be the (nu,nu) "
          "identity matrix).",
          bp::args("self", "space")))
      .add_property("ntau", &MultibodyFreeFwdDynamics::ntau,
                    "Torque dimension.")
      .add_property(
          "isUnderactuated", &MultibodyFreeFwdDynamics::isUnderactuated,
          "Whether the system is underactuated, i.e. if the actuation matrix "
          "rank is lower than the acceleration vector's dimension.")
      .add_property("actuationMatrixRank",
                    &MultibodyFreeFwdDynamics::getActuationMatrixRank,
                    "Get the rank of the actuation matrix.");

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
