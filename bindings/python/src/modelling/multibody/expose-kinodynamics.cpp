/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/python/fwd.hpp"

#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"
#include "aligator/modelling/multibody/fwd.hpp"
#include <pinocchio/multibody/model.hpp>

namespace aligator {
namespace python {

void exposeKinodynamics() {
  using namespace aligator::dynamics;
  using context::PinModel;
  using context::Scalar;
  using context::StageFunction;
  using context::StageFunctionData;
  using context::UnaryFunction;
  using ODEData = ContinuousDynamicsDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using ContinuousDynamicsAbstract = ContinuousDynamicsAbstractTpl<Scalar>;
  using KinodynamicsFwdData = KinodynamicsFwdDataTpl<Scalar>;
  using KinodynamicsFwdDynamics = KinodynamicsFwdDynamicsTpl<Scalar>;
  using context::MultibodyPhaseSpace;
  using Vector3s = typename math_types<Scalar>::Vector3s;

  const PolymorphicMultiBaseVisitor<ODEAbstract, ContinuousDynamicsAbstract>
      ode_visitor;

  bp::class_<KinodynamicsFwdDynamics, bp::bases<ODEAbstract>>(
      "KinodynamicsFwdDynamics",
      "Centroidal forward dynamics + kinematics using Pinocchio.",
      bp::init<const MultibodyPhaseSpace &, const PinModel &, const Vector3s &,
               const std::vector<bool> &,
               const std::vector<pinocchio::FrameIndex> &, const int>(
          "Constructor.", ("self"_a, "space", "model", "gravity",
                           "contact_states", "contact_ids", "force_size")))
      .def_readwrite("contact_states",
                     &KinodynamicsFwdDynamics::contact_states_)
      .def(ode_visitor);

  bp::register_ptr_to_python<shared_ptr<KinodynamicsFwdData>>();

  bp::class_<KinodynamicsFwdData, bp::bases<ODEData>>("KinodynamicsFwdData",
                                                      bp::no_init)
      .def_readwrite("pin_data", &KinodynamicsFwdData::pin_data_);
}

} // namespace python
} // namespace aligator
#endif
