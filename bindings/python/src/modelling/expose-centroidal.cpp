/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/functions.hpp"
#include "aligator/modelling/centroidal/center-of-mass-translation.hpp"
#include "aligator/modelling/centroidal/linear-momentum.hpp"
#include "aligator/modelling/centroidal/angular-momentum.hpp"

namespace aligator {
namespace python {

using context::Scalar;
using context::StageFunctionData;
using context::UnaryFunction;

void exposeCentroidalFunctions() {
  using CentroidalCoM = CentroidalCoMResidualTpl<Scalar>;
  using CentroidalCoMData = CentroidalCoMDataTpl<Scalar>;

  using LinearMomentum = LinearMomentumResidualTpl<Scalar>;
  using LinearMomentumData = LinearMomentumDataTpl<Scalar>;

  using AngularMomentum = AngularMomentumResidualTpl<Scalar>;
  using AngularMomentumData = AngularMomentumDataTpl<Scalar>;

  /*bp::class_<CentroidalCoM, bp::bases<UnaryFunction>>(
      "CentroidalCoMTranslation residual",
      "A residual function :math:`r(x) = com(x)` ",
      bp::init<const int, const int, const context::Vector3s>(
          bp::args("self", "ndx", "nu", "p_ref")))
      .def("getReference", &CentroidalCoM::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target Centroidal CoM translation.")
      .def("setReference", &CentroidalCoM::setReference,
           bp::args("self", "p_new"),
           "Set the target Centroidal CoM translation.");

  bp::register_ptr_to_python<shared_ptr<CentroidalCoMData>>();

  bp::class_<CentroidalCoMData, bp::bases<StageFunctionData>>(
      "CentroidalCoMData",
      "Data Structure for CentroidalCoM", bp::no_init);

  bp::class_<LinearMomentum, bp::bases<UnaryFunction>>(
      "LinearMomentum residual",
      "A residual function :math:`r(x) = h(x)` ",
      bp::init<const int, const int, const context::Vector3s>(
          bp::args("self", "ndx", "nu", "h_ref")))
      .def("getReference", &LinearMomentum::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target Linear Momentum.")
      .def("setReference", &LinearMomentum::setReference,
           bp::args("self", "h_new"),
           "Set the target Linear Momentum.");

  bp::register_ptr_to_python<shared_ptr<LinearMomentumData>>();

  bp::class_<LinearMomentumData, bp::bases<StageFunctionData>>(
      "LinearMomentumData",
      "Data Structure for LinearMomentum", bp::no_init);*/

  /*bp::class_<AngularMomentum, bp::bases<UnaryFunction>>(
      "AngularMomentum residual",
      "A residual function :math:`r(x) = L(x)` ",
      bp::init<const int, const int, const context::Vector3s>(
          bp::args("self", "ndx", "nu", "L_ref")))
      .def("getReference", &AngularMomentum::getReference,
           bp::args("self"), bp::return_internal_reference<>(),
           "Get the target Angular Momentum.")
      .def("setReference", &AngularMomentum::setReference,
           bp::args("self", "L_new"),
           "Set the target Angular Momentum.");

  bp::register_ptr_to_python<shared_ptr<AngularMomentumData>>();

  bp::class_<AngularMomentumData, bp::bases<StageFunctionData>>(
      "AngularMomentumData",
      "Data Structure for AngularMomentum", bp::no_init);*/
}

} // namespace python
} // namespace aligator
