/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include <proxsuite-nlp/python/polymorphic.hpp>
#include "aligator/python/visitors.hpp"
#include "aligator/python/modelling/continuous.hpp"
#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/linear-ode.hpp"

namespace aligator::python {
using context::ContinuousDynamicsAbstract;
using context::MatrixXs;
using context::ODEAbstract;
using context::ODEData;
using context::Scalar;
using context::VectorXs;
using PolyManifold = xyz::polymorphic<context::Manifold>;

void exposeODEs() {
  using dynamics::LinearODETpl;

  bp::implicitly_convertible<LinearODETpl<Scalar>,
                             xyz::polymorphic<ODEAbstract>>();
  bp::implicitly_convertible<LinearODETpl<Scalar>,
                             xyz::polymorphic<ContinuousDynamicsAbstract>>();
  bp::implicitly_convertible<PyODEAbstract<>, xyz::polymorphic<ODEAbstract>>();
  bp::implicitly_convertible<PyODEAbstract<>,
                             xyz::polymorphic<ContinuousDynamicsAbstract>>();
  proxsuite::nlp::python::register_polymorphic_to_python<
      xyz::polymorphic<ODEAbstract>>();
  bp::class_<PyODEAbstract<>, bp::bases<ContinuousDynamicsAbstract>,
             boost::noncopyable>(
      "ODEAbstract",
      "Continuous dynamics described by ordinary differential equations "
      "(ODEs).",
      bp::init<const PolyManifold &, int>(bp::args("self", "space", "nu")))
      .def("forward", bp::pure_virtual(&ODEAbstract::forward),
           bp::args("self", "x", "u", "data"),
           "Compute the value of the ODE vector field, i.e. the "
           "state time derivative :math:`\\dot{x}`.")
      .def("dForward", bp::pure_virtual(&ODEAbstract::dForward),
           bp::args("self", "x", "u", "data"),
           "Compute the derivatives of the ODE vector field with respect "
           "to the state-control pair :math:`(x, u)`.")
      .def(CreateDataPolymorphicPythonVisitor<ODEAbstract, PyODEAbstract<>>());

  bp::class_<LinearODETpl<Scalar>, bp::bases<ODEAbstract>>(
      "LinearODE",
      "Linear ordinary differential equation, :math:`\\dot{x} = Ax + Bu`.",
      bp::init<PolyManifold, MatrixXs, MatrixXs, VectorXs>(
          bp::args("self", "A", "B", "c")))
      .def(bp::init<MatrixXs, MatrixXs, VectorXs>(
          "Constructor with just the matrices; a Euclidean state space is "
          "assumed.",
          bp::args("self", "A", "B", "c")))
      .def_readonly("A", &LinearODETpl<Scalar>::A_, "State transition matrix.")
      .def_readonly("B", &LinearODETpl<Scalar>::B_, "Control matrix.")
      .def_readonly("c", &LinearODETpl<Scalar>::c_, "Constant drift term.");
}
} // namespace aligator::python
