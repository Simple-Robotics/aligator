/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/modelling/dynamics.hpp"
#include "proxddp/python/eigen_member.hpp"

#include "proxddp/modelling/linear-discrete-dynamics.hpp"

namespace proxddp {
namespace python {

void exposeExplicitDynamics();
void exposeDynamicsImplementations();

void exposeDynamics() {
  using context::DynamicsModel;
  using context::Scalar;
  using context::StageFunction;
  using ManifoldPtr = shared_ptr<context::Manifold>;

  using PyDynamicsModel = internal::PyStageFunction<DynamicsModel>;

  bp::class_<PyDynamicsModel, bp::bases<StageFunction>, boost::noncopyable>(
      "DynamicsModel",
      "Dynamics models are specific ternary functions f(x,u,x') which map "
      "to the tangent bundle of the next state variable x'.",
      bp::init<ManifoldPtr, const int, const int>(
          bp::args("self", "space", "nu", "ndx2")))
      .def(bp::init<ManifoldPtr, const int>(bp::args("self", "space", "nu")))
      .def_readonly("space", &DynamicsModel::space_)
      .def_readonly("space_next", &DynamicsModel::space_next_)
      .add_property("nx1", &DynamicsModel::nx1)
      .add_property("nx2", &DynamicsModel::nx2)
      .def(CreateDataPythonVisitor<DynamicsModel>());

  exposeExplicitDynamics();
  exposeDynamicsImplementations();
}

void exposeExplicitDynamics() {
  using context::DynamicsModel;
  using context::ExplicitDynamics;
  using context::ExplicitDynData;
  using internal::PyExplicitDynamics;

  using PyDynamicsModel = internal::PyStageFunction<DynamicsModel>;
  using ManifoldPtr = shared_ptr<context::Manifold>;

  bp::class_<PyExplicitDynamics<>, bp::bases<DynamicsModel>,
             boost::noncopyable>(
      "ExplicitDynamicsModel", "Base class for explicit dynamics.",
      bp::init<const ManifoldPtr &, const int>(
          "Constructor with state space and control dimension.",
          bp::args("self", "space", "nu")))
      .def("forward", bp::pure_virtual(&ExplicitDynamics::forward),
           bp::args("self", "x", "u", "data"),
           "Call for forward discrete dynamics.")
      .def("dForward", bp::pure_virtual(&ExplicitDynamics::dForward),
           bp::args("self", "x", "u", "data"),
           "Compute the derivatives of forward discrete dynamics.");

  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<PyDynamicsModel>>,
                                true>::expose("StdVec_Dynamics");
  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<PyExplicitDynamics<>>>,
                                true>::expose("StdVec_ExplicitDynamics");

  bp::register_ptr_to_python<shared_ptr<context::ExplicitDynData>>();

  bp::class_<ExplicitDynData, bp::bases<context::FunctionData>>(
      "ExplicitDynamicsData", "Data struct for explicit dynamics models.",
      bp::no_init)
      .add_property("dx", make_getter_eigen_ref(&ExplicitDynData::dx_))
      .add_property("xnext", make_getter_eigen_ref(&ExplicitDynData::xnext_))
      .def(PrintableVisitor<ExplicitDynData>());
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
