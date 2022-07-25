#include "proxddp/python/fwd.hpp"

#include "proxddp/modelling/dynamics/multibody-free-fwd.hpp"

namespace proxddp {
namespace python {
void exposeFreeFwdDynamics() {
  using namespace proxddp::dynamics;
  using context::Scalar;
  using ODEData = ODEDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using MultibodyFreeFwdData = MultibodyFreeFwdDataTpl<Scalar>;
  using MultibodyFreeFwdDynamics = MultibodyFreeFwdDynamicsTpl<Scalar>;

  bp::class_<MultibodyFreeFwdDynamics, bp::bases<ODEAbstract>>(
      "MultibodyFreeFwdDynamics",
      "Free-space forward dynamics on multibodies using Pinocchio's ABA "
      "algorithm.",
      bp::init<const shared_ptr<proxnlp::MultibodyPhaseSpace<Scalar>> &,
               const context::MatrixXs &>(
          bp::args("self", "space", "actuation_matrix")))
      .add_property("ntau", &MultibodyFreeFwdDynamics::ntau,
                    "Torque dimension.")
      .def(CreateDataPythonVisitor<MultibodyFreeFwdDynamics>());

  bp::register_ptr_to_python<shared_ptr<MultibodyFreeFwdData>>();

  bp::class_<MultibodyFreeFwdData, bp::bases<ODEData>>("MultibodyFreeFwdData",
                                                       bp::no_init)
      .def_readwrite("tau", &MultibodyFreeFwdData::tau_)
      .def_readwrite("dtau_du", &MultibodyFreeFwdData::dtau_du_)
      .def_readwrite("pin_data", &MultibodyFreeFwdData::pin_data_);
}
} // namespace python
} // namespace proxddp
