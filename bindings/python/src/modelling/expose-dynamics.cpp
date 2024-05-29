/// @file
/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/python/functions.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/core/dynamics.hpp"
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

using context::DynamicsModel;
using PolyManifold = xyz::polymorphic<context::Manifold>;
using context::StageFunction;

void exposeDynamicsBase() {

  using PyDynamicsModel = internal::PyStageFunction<DynamicsModel>;

  proxsuite::nlp::python::register_polymorphic_to_python<
      xyz::polymorphic<DynamicsModel>>();
  StdVectorPythonVisitor<std::vector<xyz::polymorphic<DynamicsModel>>,
                         true>::expose("StdVec_Dynamics");
  bp::class_<PyDynamicsModel, bp::bases<StageFunction>, boost::noncopyable>(
      "DynamicsModel",
      "Dynamics models are specific ternary functions f(x,u,x') which map "
      "to the tangent bundle of the next state variable x'.",
      bp::init<PolyManifold, int>(bp::args("self", "space", "nu")))
      .def(bp::init<PolyManifold, int, PolyManifold>(
          bp::args("self", "space", "nu", "space2")))
      .def_readonly("space", &DynamicsModel::space_)
      .def_readonly("space_next", &DynamicsModel::space_next_)
      .add_property("nx1", &DynamicsModel::nx1)
      .add_property("nx2", &DynamicsModel::nx2)
      .add_property("is_explicit", &DynamicsModel::is_explicit,
                    "Return whether the current model is explicit.")
      .def(
          CreateDataPolymorphicPythonVisitor<DynamicsModel, PyDynamicsModel>());
}

} // namespace python
} // namespace aligator
