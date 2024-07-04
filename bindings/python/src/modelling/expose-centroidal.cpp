/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/context.hpp"
#include "aligator/python/polymorphic-convertible.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/modelling/centroidal/centroidal-translation.hpp"
#include "aligator/modelling/centroidal/linear-momentum.hpp"
#include "aligator/modelling/centroidal/angular-momentum.hpp"
#include "aligator/modelling/centroidal/centroidal-acceleration.hpp"
#include "aligator/modelling/centroidal/friction-cone.hpp"
#include "aligator/modelling/centroidal/wrench-cone.hpp"
#include "aligator/modelling/centroidal/angular-acceleration.hpp"
#include "aligator/modelling/centroidal/centroidal-wrapper.hpp"
#include "aligator/modelling/contact-map.hpp"

namespace aligator {
namespace python {

using context::Scalar;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;
using ContactMap = ContactMapTpl<Scalar>;

void exposeContactMap() {
  bp::class_<ContactMap>(
      "ContactMap", "Store contact state and pose for centroidal problem",
      bp::init<const std::vector<bool> &,
               const StdVectorEigenAligned<context::Vector3s> &>(
          ("self"_a, "contact_states", "contact_poses")))
      .def("addContact", &ContactMap::addContact, ("self"_a, "state", "pose"),
           "Add a contact to the contact map.")
      .def("removeContact", &ContactMap::removeContact, ("self"_a, "i"),
           "Remove contact i from the contact map.")
      .add_property("size", &ContactMap::getSize, "Get map size.")
      .add_property("contact_states",
                    bp::make_function(&ContactMap::getContactStates,
                                      bp::return_internal_reference<>()),
                    "Get all the states in contact map.")
      .add_property("contact_poses",
                    bp::make_function(&ContactMap::getContactPoses,
                                      bp::return_internal_reference<>()),
                    "Get all the poses in contact map.");
}

void exposeCentroidalFunctions() {
  using CentroidalCoMResidual = CentroidalCoMResidualTpl<Scalar>;
  using CentroidalCoMData = CentroidalCoMDataTpl<Scalar>;

  using LinearMomentumResidual = LinearMomentumResidualTpl<Scalar>;
  using LinearMomentumData = LinearMomentumDataTpl<Scalar>;

  using AngularMomentumResidual = AngularMomentumResidualTpl<Scalar>;
  using AngularMomentumData = AngularMomentumDataTpl<Scalar>;

  using CentroidalAccelerationResidual =
      CentroidalAccelerationResidualTpl<Scalar>;
  using CentroidalAccelerationData = CentroidalAccelerationDataTpl<Scalar>;

  using FrictionConeResidual = FrictionConeResidualTpl<Scalar>;
  using FrictionConeData = FrictionConeDataTpl<Scalar>;

  using WrenchConeResidual = WrenchConeResidualTpl<Scalar>;
  using WrenchConeData = WrenchConeDataTpl<Scalar>;

  using AngularAccelerationResidual = AngularAccelerationResidualTpl<Scalar>;
  using AngularAccelerationData = AngularAccelerationDataTpl<Scalar>;

  using CentroidalWrapperResidual = CentroidalWrapperResidualTpl<Scalar>;
  using CentroidalWrapperData = CentroidalWrapperDataTpl<Scalar>;

  PolymorphicMultiBaseVisitor<StageFunction> func_visitor;
  PolymorphicMultiBaseVisitor<UnaryFunction, StageFunction> unary_visitor;

  bp::class_<CentroidalCoMResidual, bp::bases<UnaryFunction>>(
      "CentroidalCoMResidual", "A residual function :math:`r(x) = com(x)` ",
      bp::init<const int, const int, const context::Vector3s &>(
          ("self"_a, "ndx", "nu", "p_ref")))
      .def("getReference", &CentroidalCoMResidual::getReference, ("self"_a),
           bp::return_internal_reference<>(),
           "Get the target Centroidal CoM translation.")
      .def("setReference", &CentroidalCoMResidual::setReference,
           ("self"_a, "p_new"), "Set the target Centroidal CoM translation.")
      .def(unary_visitor);

  bp::register_ptr_to_python<shared_ptr<CentroidalCoMData>>();

  bp::class_<CentroidalCoMData, bp::bases<StageFunctionData>>(
      "CentroidalCoMData", "Data Structure for CentroidalCoM", bp::no_init);

  bp::class_<LinearMomentumResidual, bp::bases<UnaryFunction>>(
      "LinearMomentumResidual", "A residual function :math:`r(x) = h(x)` ",
      bp::init<const int, const int, const context::Vector3s &>(
          ("self"_a, "ndx", "nu", "h_ref")))
      .def("getReference", &LinearMomentumResidual::getReference, ("self"_a),
           bp::return_internal_reference<>(), "Get the target Linear Momentum.")
      .def("setReference", &LinearMomentumResidual::setReference,
           ("self"_a, "h_new"), "Set the target Linear Momentum.")
      .def(unary_visitor);

  bp::register_ptr_to_python<shared_ptr<LinearMomentumData>>();

  bp::class_<LinearMomentumData, bp::bases<StageFunctionData>>(
      "LinearMomentumData", "Data Structure for LinearMomentum", bp::no_init);

  bp::class_<AngularMomentumResidual, bp::bases<UnaryFunction>>(
      "AngularMomentumResidual", "A residual function :math:`r(x) = L(x)` ",
      bp::init<const int, const int, const context::Vector3s &>(
          ("self"_a, "ndx", "nu", "L_ref")))
      .def("getReference", &AngularMomentumResidual::getReference, "self"_a,
           bp::return_internal_reference<>(),
           "Get the target Angular Momentum.")
      .def("setReference", &AngularMomentumResidual::setReference,
           ("self"_a, "L_new"), "Set the target Angular Momentum.")
      .def(unary_visitor);

  bp::register_ptr_to_python<shared_ptr<AngularMomentumData>>();

  bp::class_<AngularMomentumData, bp::bases<StageFunctionData>>(
      "AngularMomentumData", "Data Structure for AngularMomentum", bp::no_init);

  bp::class_<CentroidalAccelerationResidual, bp::bases<StageFunction>>(
      "CentroidalAccelerationResidual",
      "A residual function :math:`r(x) = cddot(x)` ",
      bp::init<const int, const int, const double, const context::Vector3s &,
               const ContactMap &, const int>(("self"_a, "ndx", "nu", "mass",
                                               "gravity", "contact_map",
                                               "force_size")))
      .def_readwrite("contact_map",
                     &CentroidalAccelerationResidual::contact_map_)
      .def(CreateDataPythonVisitor<CentroidalAccelerationResidual>())
      .def(func_visitor);

  bp::register_ptr_to_python<shared_ptr<CentroidalAccelerationData>>();

  bp::class_<CentroidalAccelerationData, bp::bases<StageFunctionData>>(
      "CentroidalAccelerationData", "Data Structure for CentroidalAcceleration",
      bp::no_init);

  bp::class_<FrictionConeResidual, bp::bases<StageFunction>>(
      "FrictionConeResidual",
      "A residual function :math:`r(x) = [fz, mu2 * fz2 - (fx2 + fy2)]` ",
      bp::init<const int, const int, const int, const double, const double>(
          ("self"_a, "ndx", "nu", "k", "mu", "epsilon")))
      .def(func_visitor);

  bp::register_ptr_to_python<shared_ptr<FrictionConeData>>();

  bp::class_<FrictionConeData, bp::bases<StageFunctionData>>(
      "FrictionConeData", "Data Structure for FrictionCone", bp::no_init);

  bp::class_<WrenchConeResidual, bp::bases<StageFunction>>(
      "WrenchConeResidual",
      "A residual function :math:`r(x) = [fz, mu2 * fz2 - (fx2 + fy2)]` ",
      bp::init<const int, const int, const int, const double, const double,
               const double>(("self"_a, "ndx", "nu", "k", "mu", "L", "W")))
      .def(func_visitor);

  bp::register_ptr_to_python<shared_ptr<WrenchConeData>>();

  bp::class_<WrenchConeData, bp::bases<StageFunctionData>>(
      "WrenchConeData", "Data Structure for WrenchCone", bp::no_init);

  bp::class_<AngularAccelerationResidual, bp::bases<StageFunction>>(
      "AngularAccelerationResidual",
      "A residual function :math:`r(x) = Ldot(x)` ",
      bp::init<const int, const int, const double, const context::Vector3s &,
               const ContactMap &, const int>(("self"_a, "ndx", "nu", "mass",
                                               "gravity", "contact_map",
                                               "force_size")))
      .def_readwrite("contact_map", &AngularAccelerationResidual::contact_map_)
      .def(CreateDataPythonVisitor<AngularAccelerationResidual>())
      .def(func_visitor);

  bp::register_ptr_to_python<shared_ptr<AngularAccelerationData>>();

  bp::class_<AngularAccelerationData, bp::bases<StageFunctionData>>(
      "AngularAccelerationData", "Data Structure for AngularAcceleration",
      bp::no_init);

  bp::class_<CentroidalWrapperResidual, bp::bases<UnaryFunction>>(
      "CentroidalWrapperResidual",
      "A wrapper for centroidal cost with smooth control",
      bp::init<xyz::polymorphic<StageFunction>>(("self"_a, "centroidal_cost")))
      .def_readwrite("centroidal_cost",
                     &CentroidalWrapperResidual::centroidal_cost_)
      .def(CreateDataPythonVisitor<CentroidalWrapperResidual>())
      .def(unary_visitor);

  bp::register_ptr_to_python<shared_ptr<CentroidalWrapperData>>();

  bp::class_<CentroidalWrapperData, bp::bases<StageFunctionData>>(
      "CentroidalWrapperData", "Data Structure for CentroidalWrapper",
      bp::no_init);
}

} // namespace python
} // namespace aligator
