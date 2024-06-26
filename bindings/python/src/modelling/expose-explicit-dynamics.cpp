/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/polymorphic-convertible.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/python/modelling/explicit-dynamics.hpp"
#include "aligator/modelling/linear-discrete-dynamics.hpp"

namespace aligator {
namespace python {

using context::DynamicsModel;
using context::ExplicitDynamics;
using context::ExplicitDynamicsData;
using context::Scalar;
using ManifoldPtr = xyz::polymorphic<context::Manifold>;
PolymorphicMultiBaseVisitor<DynamicsModel, ExplicitDynamics>
    exp_dynamics_visitor;

// fwd declaration
void exposeExplicitBase();
void exposeLinearDiscreteDynamics();
// fwd declaration, see expose-direct-sum.cpp
void exposeExplicitDynDirectSum();

//// impl

void exposeExplicitDynamics() {
  exposeExplicitBase();
  exposeLinearDiscreteDynamics();
  exposeExplicitDynDirectSum();
}

struct ExplicitDataWrapper : ExplicitDynamicsData,
                             bp::wrapper<ExplicitDynamicsData> {
  using ExplicitDynamicsData::ExplicitDynamicsData;
};

void exposeExplicitBase() {

  using PolyExplicitDynamics = xyz::polymorphic<ExplicitDynamics>;
  StdVectorPythonVisitor<std::vector<PolyExplicitDynamics>, true>::expose(
      "StdVec_ExplicitDynamics");

  register_polymorphic_to_python<PolyExplicitDynamics>();

  bp::class_<PyExplicitDynamics<>, bp::bases<DynamicsModel>,
             boost::noncopyable>(
      "ExplicitDynamicsModel", "Base class for explicit dynamics.",
      bp::init<ManifoldPtr, const int>(
          "Constructor with state space and control dimension.",
          ("self"_a, "space", "nu"))[bp::with_custodian_and_ward<1, 2>()])
      .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
           ("self"_a, "x", "u", "data"), "Call for forward discrete dynamics.")
      .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
           ("self"_a, "x", "u", "data"),
           "Compute the derivatives of forward discrete dynamics.")
      .def(exp_dynamics_visitor)
      .def(CreateDataPolymorphicPythonVisitor<ExplicitDynamics,
                                              PyExplicitDynamics<>>());

  bp::register_ptr_to_python<shared_ptr<ExplicitDynamicsData>>();

  bp::class_<ExplicitDataWrapper, bp::bases<context::StageFunctionData>,
             boost::noncopyable>(
      "ExplicitDynamicsData", "Data struct for explicit dynamics models.",
      bp::init<int, int, int, int>(("self"_a, "ndx1", "nu", "nx2", "ndx2")))
      .add_property(
          "xnext",
          bp::make_getter(&ExplicitDynamicsData::xnext_ref,
                          bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "dx", bp::make_getter(&ExplicitDynamicsData::dx_ref,
                                bp::return_value_policy<bp::return_by_value>()))
      .def(PrintableVisitor<ExplicitDynamicsData>());
}

void exposeLinearDiscreteDynamics() {
  using context::MatrixXs;
  using context::VectorXs;
  using namespace aligator::dynamics;

  bp::class_<LinearDiscreteDynamicsTpl<Scalar>,
             bp::bases<context::ExplicitDynamics>>(
      "LinearDiscreteDynamics",
      "Linear discrete dynamics :math:`x[t+1] = Ax[t] + Bu[t] + c[t]` in "
      "Euclidean space, or "
      "on the tangent state space.",
      bp::init<const MatrixXs &, const MatrixXs &, const VectorXs &>(
          ("self"_a, "A", "B", "c")))
      .def_readonly("A", &LinearDiscreteDynamicsTpl<Scalar>::A_)
      .def_readonly("B", &LinearDiscreteDynamicsTpl<Scalar>::B_)
      .def_readonly("c", &LinearDiscreteDynamicsTpl<Scalar>::c_)
      .def(exp_dynamics_visitor);
}

} // namespace python
} // namespace aligator
