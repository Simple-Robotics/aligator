/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, 2023-2025 INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/python/modelling/explicit-dynamics.hpp"
#include "aligator/modelling/linear-discrete-dynamics.hpp"

namespace aligator {
namespace python {

using context::DynamicsData;
using context::DynamicsModel;
using context::ExplicitDynamics;
using context::ExplicitDynamicsData;
using context::Scalar;
using PolyManifold = xyz::polymorphic<context::Manifold>;
static PolymorphicMultiBaseVisitor<ExplicitDynamics> exp_dynamics_visitor;

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
      "StdVec_ExplicitDynamics",
      eigenpy::details::overload_base_get_item_for_std_vector<
          std::vector<PolyExplicitDynamics>>{});

  register_polymorphic_to_python<PolyExplicitDynamics>();

  bp::class_<PyExplicitDynamics<>, boost::noncopyable>(
      "ExplicitDynamicsModel", "Base class for explicit dynamics.",
      bp::init<const PolyManifold &, const int>(
          "Constructor with state space and control dimension.",
          ("self"_a, "space", "nu")))
      .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
           ("self"_a, "x", "u", "data"), "Call for forward discrete dynamics.")
      .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
           ("self"_a, "x", "u", "data"),
           "Compute the derivatives of forward discrete dynamics.")
      .def(CreateDataPolymorphicPythonVisitor<ExplicitDynamics,
                                              PyExplicitDynamics<>>());

  bp::register_ptr_to_python<shared_ptr<ExplicitDynamicsData>>();

  bp::class_<ExplicitDataWrapper, boost::noncopyable>(
      "ExplicitDynamicsData", "Data struct for explicit dynamics models.",
      bp::no_init)
      .def_readwrite("xnext", &ExplicitDataWrapper::xnext_)
      .def_readwrite("dx", &ExplicitDataWrapper::dx_)
      .def_readwrite("jac_buffer", &ExplicitDataWrapper::jac_buffer_)
      .add_property(
          "Jx",
          +[](ExplicitDataWrapper &d) -> context::MatrixRef { return d.Jx(); })
      .add_property(
          "Ju",
          +[](ExplicitDataWrapper &d) -> context::MatrixRef { return d.Ju(); })
      .def(bp::init<const ExplicitDynamics &>(("self"_a, "model")))
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
