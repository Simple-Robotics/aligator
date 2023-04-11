/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/modelling/explicit-dynamics.hpp"
#include "proxddp/python/eigen-member.hpp"
#include "proxddp/python/visitors.hpp"

#include "proxddp/modelling/linear-discrete-dynamics.hpp"

namespace proxddp {
namespace python {

void exposeDynamicsBase();
void exposeExplicitDynamics();
void exposeDynamicsImplementations();

void exposeDynamics() {
  exposeDynamicsBase();
  exposeExplicitDynamics();
  exposeDynamicsImplementations();
}

using context::DynamicsModel;
using context::ExplicitDynamics;
using context::ExplicitDynData;
using context::Manifold;
using context::Scalar;
using ManifoldPtr = shared_ptr<context::Manifold>;
using context::StageFunction;
using internal::PyExplicitDynamics;

void exposeDynamicsBase() {

  using PyDynamicsModel = internal::PyStageFunction<DynamicsModel>;

  StdVectorPythonVisitor<std::vector<shared_ptr<DynamicsModel>>, true>::expose(
      "StdVec_Dynamics");

  bp::class_<PyDynamicsModel, bp::bases<StageFunction>, boost::noncopyable>(
      "DynamicsModel",
      "Dynamics models are specific ternary functions f(x,u,x') which map "
      "to the tangent bundle of the next state variable x'.",
      bp::init<ManifoldPtr, int>(bp::args("self", "space", "nu")))
      .def(bp::init<ManifoldPtr, int, ManifoldPtr>(
          bp::args("self", "space", "nu", "space2")))
      .def_readonly("space", &DynamicsModel::space_)
      .def_readonly("space_next", &DynamicsModel::space_next_)
      .add_property("nx1", &DynamicsModel::nx1)
      .add_property("nx2", &DynamicsModel::nx2)
      .add_property("is_explicit", &DynamicsModel::is_explicit,
                    "Return whether the current model is explicit.");
}

void exposeExplicitDynamics() {

  StdVectorPythonVisitor<std::vector<shared_ptr<ExplicitDynamics>>,
                         true>::expose("StdVec_ExplicitDynamics");

  bp::class_<PyExplicitDynamics<>, bp::bases<DynamicsModel>,
             boost::noncopyable>(
      "ExplicitDynamicsModel", "Base class for explicit dynamics.",
      bp::init<ManifoldPtr, const int>(
          "Constructor with state space and control dimension.",
          bp::args("self", "space", "nu")))
      .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
           bp::args("self", "x", "u", "data"),
           "Call for forward discrete dynamics.")
      .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
           bp::args("self", "x", "u", "data"),
           "Compute the derivatives of forward discrete dynamics.")
      .def(CreateDataPolymorphicPythonVisitor<ExplicitDynamics,
                                              PyExplicitDynamics<>>());

  bp::register_ptr_to_python<shared_ptr<context::ExplicitDynData>>();

  bp::class_<ExplicitDynData, bp::bases<context::FunctionData>,
             boost::noncopyable>("ExplicitDynamicsData",
                                 "Data struct for explicit dynamics models.",
                                 bp::init<int, int, int, int>(bp::args(
                                     "self", "ndx1", "nu", "nx2", "ndx2")))
      .add_property(
          "xnext",
          bp::make_getter(&ExplicitDynData::xnext_ref,
                          bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "dx", bp::make_getter(&ExplicitDynData::dx_ref,
                                bp::return_value_policy<bp::return_by_value>()))
      .def(PrintableVisitor<ExplicitDynData>());
}

void exposeDynamicsImplementations() {
  using context::MatrixXs;
  using context::VectorXs;
  using namespace proxddp::dynamics;

  bp::class_<LinearDiscreteDynamicsTpl<Scalar>,
             bp::bases<context::ExplicitDynamics>>(
      "LinearDiscreteDynamics",
      "Linear discrete dynamics :math:`x[t+1] = Ax[t] + Bu[t] + c[t]` in "
      "Euclidean space, or "
      "on the tangent state space.",
      bp::init<const MatrixXs &, const MatrixXs &, const VectorXs &>(
          bp::args("self", "A", "B", "c")))
      .def_readonly("A", &LinearDiscreteDynamicsTpl<Scalar>::A_)
      .def_readonly("B", &LinearDiscreteDynamicsTpl<Scalar>::B_)
      .def_readonly("c", &LinearDiscreteDynamicsTpl<Scalar>::c_);
}

} // namespace python
} // namespace proxddp
