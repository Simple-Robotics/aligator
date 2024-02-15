/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>
#include <pinocchio/multibody/model.hpp>

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
  using KinodynamicsFwdData = KinodynamicsFwdDataTpl<Scalar>;
  using KinodynamicsFwdDynamics = KinodynamicsFwdDynamicsTpl<Scalar>;
  using proxsuite::nlp::CartesianProductTpl;
  using Vector3s = typename math_types<Scalar>::Vector3s;
  using ContactMap = ContactMapTpl<Scalar>;

  using StateManifoldPtr = shared_ptr<CartesianProductTpl<Scalar>>;
  using Model = pinocchio::ModelTpl<Scalar>;

  bp::class_<KinodynamicsFwdDynamics, bp::bases<ODEAbstract>>(
      "KinodynamicsFwdDynamics",
      "Centroidal forward dynamics + kinematics using Pinocchio.",
      bp::init<const StateManifoldPtr &, const Model &, const Vector3s &,
               const ContactMap &>(
          "Constructor.", bp::args("self", "space", "gravity", "contact_map")));

  bp::register_ptr_to_python<shared_ptr<KinodynamicsFwdData>>();

  bp::class_<KinodynamicsFwdData, bp::bases<ODEData>>("KinodynamicsFwdData",
                                                      bp::no_init)
      .def_readwrite("pin_data", &KinodynamicsFwdData::pin_data_);
}
} // namespace python
} // namespace aligator
