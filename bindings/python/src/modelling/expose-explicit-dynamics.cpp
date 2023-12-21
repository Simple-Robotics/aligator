/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/modelling/explicit-dynamics.hpp"

#include "aligator/modelling/linear-discrete-dynamics.hpp"

namespace aligator {
namespace python {

using context::DynamicsModel;
using context::ExplicitDynamics;
using context::ExplicitDynamicsData;
using context::Scalar;
using ManifoldPtr = shared_ptr<context::Manifold>;
using internal::PyExplicitDynamics;

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

  bp::register_ptr_to_python<shared_ptr<ExplicitDynamicsData>>();

  bp::class_<ExplicitDataWrapper, bp::bases<context::StageFunctionData>,
             boost::noncopyable>("ExplicitDynamicsData",
                                 "Data struct for explicit dynamics models.",
                                 bp::init<int, int, int, int>(bp::args(
                                     "self", "ndx1", "nu", "nx2", "ndx2")))
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
          bp::args("self", "A", "B", "c")))
      .def_readonly("A", &LinearDiscreteDynamicsTpl<Scalar>::A_)
      .def_readonly("B", &LinearDiscreteDynamicsTpl<Scalar>::B_)
      .def_readonly("c", &LinearDiscreteDynamicsTpl<Scalar>::c_);
}

} // namespace python
} // namespace aligator
