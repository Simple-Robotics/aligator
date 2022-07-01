#include "proxddp/python/modelling/dynamics.hpp"

#include "proxddp/modelling/linear-discrete-dynamics.hpp"

namespace proxddp {
namespace python {

void exposeDynamicsImplementations();

void exposeDynamics() {
  using context::DynamicsModel;
  using context::Manifold;
  using context::Scalar;
  using context::StageFunction;
  using ManifoldPtr = shared_ptr<context::Manifold>;
  using context::ExplicitDynamics;
  using internal::PyExplicitDynamics;
  using internal::PyStageFunction;

  using PyDynamicsModel = PyStageFunction<DynamicsModel>;

  bp::class_<PyDynamicsModel, bp::bases<StageFunction>, boost::noncopyable>(
      "DynamicsModel",
      "Dynamics models are specific ternary functions f(x,u,x') which map "
      "to the tangent bundle of the next state variable x'.",
      bp::init<const int, const int, const int>(
          bp::args("self", "ndx1", "nu", "ndx2")))
      .def(bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def(CreateDataPythonVisitor<DynamicsModel>());

  bp::class_<PyExplicitDynamics<>, bp::bases<DynamicsModel>,
             boost::noncopyable>(
      "ExplicitDynamicsModel", "Base class for explicit dynamics.",
      bp::init<const int, const int, const ManifoldPtr &>(
          "Construtor with current state dimension, control dimension and "
          "output state space.",
          bp::args("self", "ndx1", "nu", "next_space")))
      .def(bp::init<const ManifoldPtr &, const int>(
          "Constructor with state space and control dimension.",
          bp::args("self", "space", "nu")))
      .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
           bp::args("self", "x", "u", "data"),
           "Call for forward discrete dynamics.")
      .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
           bp::args("self", "x", "u", "data"),
           "Compute the derivatives of forward discrete dynamics.")
      .add_property("space",
                    bp::make_function(&ExplicitDynamics::out_space,
                                      bp::return_internal_reference<>()),
                    "Output space.")
      .def(CreateDataPythonVisitor<ExplicitDynamics>());

  bp::register_ptr_to_python<shared_ptr<context::ExplicitDynData>>();

  bp::class_<context::ExplicitDynData, bp::bases<context::StageFunctionData>>(
      "ExplicitDynamicsData", "Data struct for explicit dynamics models.",
      bp::no_init)
      .add_property(
          "dx", bp::make_getter(&context::ExplicitDynData::dxref_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "xout",
          bp::make_getter(&context::ExplicitDynData::xoutref_,
                          bp::return_value_policy<bp::return_by_value>()));

  exposeDynamicsImplementations();
}

void exposeDynamicsImplementations() {
  using context::MatrixXs;
  using context::Scalar;
  using context::VectorXs;
  using namespace proxddp::dynamics;

  bp::class_<LinearDiscreteDynamicsTpl<Scalar>,
             bp::bases<context::ExplicitDynamics>>(
      "LinearDiscreteDynamics",
      "Linear discrete dynamics x[t+1] = Ax[t] + Bu[t] in Euclidean space, or "
      "on the tangent state space.",
      bp::init<const MatrixXs &, const MatrixXs &, const VectorXs &>(
          bp::args("self", "A", "B", "c")))
      .def_readonly("A", &LinearDiscreteDynamicsTpl<Scalar>::A_)
      .def_readonly("B", &LinearDiscreteDynamicsTpl<Scalar>::B_)
      .def_readonly("c", &LinearDiscreteDynamicsTpl<Scalar>::c_);
}

} // namespace python
} // namespace proxddp
