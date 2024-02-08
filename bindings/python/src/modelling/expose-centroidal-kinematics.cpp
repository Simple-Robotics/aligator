/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"

#include "aligator/modelling/dynamics/centroidal-kinematics-fwd.hpp"
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>
#include <eigenpy/std-pair.hpp>

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
  using CentroidalPtr = shared_ptr<CentroidalFwdDynamicsTpl<Scalar>>;
  using proxsuite::nlp::CartesianProductTpl;
  using Vector3s = typename math_types<Scalar>::Vector3s;

  using StateManifoldPtr = shared_ptr<CartesianProductTpl<Scalar>>;

  bp::class_<CentroidalKinematicsFwdDynamics, bp::bases<ODEAbstract>>(
      "CentroidalKinematicsFwdDynamics",
      "Centroidal forward dynamics + kinematics using Pinocchio.",
      bp::init<StateManifoldPtr, const size_t &, CentroidalPtr>(
          "Constructor.", bp::args("self", "space", "nv", "centroidal")));

  bp::register_ptr_to_python<shared_ptr<CentroidalKinematicsFwdData>>();

  bp::class_<CentroidalKinematicsFwdData, bp::bases<ODEData>>(
      "CentroidalKinematicsFwdData", bp::no_init)
      .def_readwrite("centroidal_data",
                     &CentroidalKinematicsFwdData::centroidal_data_);

  eigenpy::StdPairConverter<std::pair<bool, Vector3s>>::registration();
  StdVectorPythonVisitor<std::vector<std::pair<bool, Vector3s>>, true>::expose(
      "StdVec_StdPair_map");
}
} // namespace python
} // namespace aligator
