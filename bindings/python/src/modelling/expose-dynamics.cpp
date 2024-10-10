/// @file
/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/python/dynamics.hpp"
#include "aligator/python/eigen-member.hpp"
#include "aligator/python/visitors.hpp"
#include <proxsuite-nlp/manifold-base.hpp>

namespace aligator {
namespace python {

// fwd declaration
void exposeDynamicsBase();
// fwd declaration, see expose-explicit-dynamics.cpp
void exposeExplicitDynamics();

void exposeDynamics() {
  exposeDynamicsBase();
  exposeExplicitDynamics();
}

using context::DynamicsData;
using context::DynamicsModel;
using PolyManifold = xyz::polymorphic<context::Manifold>;

void exposeDynamicsBase() {

  using PyDynamicsModel = PyDynamics<DynamicsModel>;
  using PolyDynamicsModel = xyz::polymorphic<DynamicsModel>;

  register_polymorphic_to_python<PolyDynamicsModel>();
  StdVectorPythonVisitor<std::vector<PolyDynamicsModel>, true>::expose(
      "StdVec_Dynamics",
      eigenpy::details::overload_base_get_item_for_std_vector<
          std::vector<PolyDynamicsModel>>{});
  bp::class_<PyDynamicsModel, boost::noncopyable>(
      "DynamicsModel",
      "Dynamics models are specific ternary functions f(x,u,x') which map "
      "to the tangent bundle of the next state variable x'.",
      bp::init<PolyManifold, int>(("self"_a, "space", "nu")))
      .def(bp::init<PolyManifold, int, PolyManifold>(
          bp::args("self", "space", "nu", "space_next")))
      .def_readonly("space", &DynamicsModel::space_)
      .def_readonly("space_next", &DynamicsModel::space_next_)
      .add_property("nx1", &DynamicsModel::nx1)
      .add_property("nx2", &DynamicsModel::nx2)
      .def_readonly("ndx1", &DynamicsModel::ndx1)
      .def_readonly("nu", &DynamicsModel::nu)
      .def_readonly("ndx2", &DynamicsModel::ndx2)
      .add_property("isExplicit", &DynamicsModel::isExplicit,
                    "Return whether the current model is explicit.")
      .def("evaluate", &DynamicsModel::evaluate,
           ("self"_a, "x", "u", "y", "data"))
      .def("computeJacobians", &DynamicsModel::computeJacobians,
           ("self"_a, "x", "u", "y", "data"))
      .def("computeVectorHessianProducts",
           &DynamicsModel::computeVectorHessianProducts,
           ("self"_a, "x", "u", "y", "lbda", "data"))
      .def(CreateDataPolymorphicPythonVisitor<DynamicsModel, PyDynamicsModel>())
      .def(PolymorphicMultiBaseVisitor<DynamicsModel>())
      .enable_pickling_(true);

  bp::register_ptr_to_python<shared_ptr<DynamicsData>>();
  bp::class_<DynamicsData, boost::noncopyable>("DynamicsData", bp::no_init)
      .def(bp::init<const DynamicsModel &>(("self"_a, "model")))
      .add_property(
          "value",
          bp::make_getter(&DynamicsData::valref_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Function value.")
      .add_property("jac_buffer",
                    make_getter_eigen_matrix(&DynamicsData::jac_buffer_),
                    "Buffer of the full function Jacobian wrt (x,u,y).")
      .add_property(
          "Jx",
          bp::make_getter(&DynamicsData::Jx_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $x$.")
      .add_property(
          "Ju",
          bp::make_getter(&DynamicsData::Ju_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $u$.")
      .add_property(
          "Jy",
          bp::make_getter(&DynamicsData::Jy_,
                          bp::return_value_policy<bp::return_by_value>()),
          "Jacobian with respect to $y$.")
      .add_property("Hxx", make_getter_eigen_matrix(&DynamicsData::Hxx_),
                    "Hessian with respect to $(x, x)$.")
      .add_property("Hxu", make_getter_eigen_matrix(&DynamicsData::Hxu_),
                    "Hessian with respect to $(x, u)$.")
      .add_property("Hxy", make_getter_eigen_matrix(&DynamicsData::Hxy_),
                    "Hessian with respect to $(x, y)$.")
      .add_property("Huu", make_getter_eigen_matrix(&DynamicsData::Huu_),
                    "Hessian with respect to $(u, u)$.")
      .add_property("Huy", make_getter_eigen_matrix(&DynamicsData::Huy_),
                    "Hessian with respect to $(x, y)$.")
      .add_property("Hyy", make_getter_eigen_matrix(&DynamicsData::Hyy_),
                    "Hessian with respect to $(y, y)$.")
      // .def(PrintableVisitor<DynamicsData>())
      .def(PrintAddressVisitor<DynamicsData>());
}

} // namespace python
} // namespace aligator
