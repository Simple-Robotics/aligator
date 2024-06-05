/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

namespace aligator {
namespace python {
void exposeFreeFwdDynamics() {
  using namespace aligator::dynamics;
  using context::Scalar;
  using ODEData = ODEDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using MultibodyFreeFwdData = MultibodyFreeFwdDataTpl<Scalar>;
  using MultibodyFreeFwdDynamics = MultibodyFreeFwdDynamicsTpl<Scalar>;
  using proxsuite::nlp::MultibodyPhaseSpace;

  using StateManifoldPtr = shared_ptr<MultibodyPhaseSpace<Scalar>>;

  bp::class_<MultibodyFreeFwdDynamics, bp::bases<ODEAbstract>>(
      "MultibodyFreeFwdDynamics",
      "Free-space forward dynamics on multibodies using Pinocchio's ABA "
      "algorithm.",
      bp::init<StateManifoldPtr, const context::MatrixXs &>(
          "Constructor where the actuation matrix is provided.",
          bp::args("self", "space", "actuation_matrix")))
      .def(bp::init<StateManifoldPtr>(
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
      .add_property("multibody_data",
                    bp::make_getter(&MultibodyFreeFwdData::multibody_data_,
                                    bp::return_internal_reference<>()));
}
} // namespace python
} // namespace aligator
