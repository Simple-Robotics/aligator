/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/centroidal-kinematics-fwd.hpp"
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>
#include <pinocchio/multibody/model.hpp>

namespace aligator {
namespace python {
void exposeCentroidalKinematicsDynamics() {
  using namespace aligator::dynamics;
  using context::Scalar;
  using ODEData = ODEDataTpl<Scalar>;
  using ODEAbstract = ODEAbstractTpl<Scalar>;
  using CentroidalKinematicsFwdData = CentroidalKinematicsFwdDataTpl<Scalar>;
  using CentroidalKinematicsFwdDynamics =
      CentroidalKinematicsFwdDynamicsTpl<Scalar>;
  using proxsuite::nlp::CartesianProductTpl;
  using Vector3s = typename math_types<Scalar>::Vector3s;
  using ContactMap = ContactMapTpl<Scalar>;

  using StateManifoldPtr = shared_ptr<CartesianProductTpl<Scalar>>;
  using Model = pinocchio::ModelTpl<Scalar>;

  bp::class_<CentroidalKinematicsFwdDynamics, bp::bases<ODEAbstract>>(
      "CentroidalKinematicsFwdDynamics",
      "Centroidal forward dynamics + kinematics using Pinocchio.",
      bp::init<const StateManifoldPtr &, const Model &, const Vector3s &,
               const ContactMap &>(
          "Constructor.", bp::args("self", "space", "gravity", "contact_map")));

  bp::register_ptr_to_python<shared_ptr<CentroidalKinematicsFwdData>>();

  bp::class_<CentroidalKinematicsFwdData, bp::bases<ODEData>>(
      "CentroidalKinematicsFwdData", bp::no_init)
      .def_readwrite("pin_data", &CentroidalKinematicsFwdData::pin_data_);
}
} // namespace python
} // namespace aligator
