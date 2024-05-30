/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"
#include <pinocchio/multibody/fwd.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/model.hpp>
#include "aligator/python/polymorphic-convertible.hpp"

namespace aligator {
namespace python {
void exposeKinodynamics() {
  using namespace aligator::dynamics;
  using context::Scalar;
  using context::StageFunction;
  using context::StageFunctionData;
  using context::UnaryFunction;
  using ODEData = ODEDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using ContinuousDynamicsAbstract = ContinuousDynamicsAbstractTpl<Scalar>;
  using KinodynamicsFwdData = KinodynamicsFwdDataTpl<Scalar>;
  using KinodynamicsFwdDynamics = KinodynamicsFwdDynamicsTpl<Scalar>;
  using Manifold = proxsuite::nlp::MultibodyPhaseSpace<Scalar>;
  using Vector3s = typename math_types<Scalar>::Vector3s;

  using Model = pinocchio::ModelTpl<Scalar>;

  convertibleToPolymorphicBases<KinodynamicsFwdDynamics,
                                ContinuousDynamicsAbstract, ODEAbstract>();
  bp::class_<KinodynamicsFwdDynamics, bp::bases<ODEAbstract>>(
      "KinodynamicsFwdDynamics",
      "Centroidal forward dynamics + kinematics using Pinocchio.",
      bp::init<const Manifold &, const Model &, const Vector3s &,
               const std::vector<bool> &,
               const std::vector<pinocchio::FrameIndex> &, const int>(
          "Constructor.",
          bp::args("self", "space", "model", "gravity", "contact_states",
                   "contact_ids", "force_size")))
      .def_readwrite("contact_states",
                     &KinodynamicsFwdDynamics::contact_states_);

  bp::register_ptr_to_python<shared_ptr<KinodynamicsFwdData>>();

  bp::class_<KinodynamicsFwdData, bp::bases<ODEData>>("KinodynamicsFwdData",
                                                      bp::no_init)
      .def_readwrite("pin_data", &KinodynamicsFwdData::pin_data_);
}
} // namespace python
} // namespace aligator
