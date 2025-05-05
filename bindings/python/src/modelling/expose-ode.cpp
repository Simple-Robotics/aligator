/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/python/polymorphic-convertible.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/python/modelling/continuous.hpp"
#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/linear-ode.hpp"
#include "aligator/modelling/dynamics/centroidal-fwd.hpp"
#include "aligator/modelling/dynamics/continuous-centroidal-fwd.hpp"
#include "aligator/modelling/contact-map.hpp"

namespace aligator::python {
using namespace ::aligator::dynamics;
using context::ContinuousDynamicsAbstract;
using context::MatrixXs;
using context::ODEAbstract;
using context::ODEData;
using context::Scalar;
using context::VectorSpace;
using context::VectorXs;
using PolyManifold = xyz::polymorphic<context::Manifold>;

using CentroidalFwdDynamics = CentroidalFwdDynamicsTpl<Scalar>;
using ContinuousCentroidalFwdDynamics =
    ContinuousCentroidalFwdDynamicsTpl<Scalar>;
using Vector3s = typename math_types<Scalar>::Vector3s;
using ContactMap = ContactMapTpl<Scalar>;

void exposeODEs() {
  register_polymorphic_to_python<xyz::polymorphic<ODEAbstract>>();
  PolymorphicMultiBaseVisitor<ODEAbstract, ContinuousDynamicsAbstract>
      ode_visitor;

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
      .def(CreateDataPolymorphicPythonVisitor<ODEAbstract, PyODEAbstract<>>())
      .def(ode_visitor);

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
      .def_readonly("c", &LinearODETpl<Scalar>::c_, "Constant drift term.")
      .def(ode_visitor);

  bp::class_<CentroidalFwdDynamics, bp::bases<ODEAbstract>>(
      "CentroidalFwdDynamics",
      "Nonlinear centroidal dynamics with preplanned feet positions",
      bp::init<const VectorSpace &, const double, const Vector3s &,
               const ContactMap &, const int>(
          bp::args("self", "space", "total mass", "gravity", "contact_map",
                   "force_size")))
      .def_readwrite("contact_map", &CentroidalFwdDynamics::contact_map_)
      .def(CreateDataPythonVisitor<CentroidalFwdDynamics>())
      .def(ode_visitor);

  bp::register_ptr_to_python<shared_ptr<CentroidalFwdDataTpl<Scalar>>>();
  bp::class_<CentroidalFwdDataTpl<Scalar>, bp::bases<ODEData>>(
      "CentroidalFwdData", bp::no_init);

  bp::class_<ContinuousCentroidalFwdDynamics, bp::bases<ODEAbstract>>(
      "ContinuousCentroidalFwdDynamics",
      "Nonlinear centroidal dynamics with preplanned feet positions and smooth "
      "forces",
      bp::init<const VectorSpace &, const double, const Vector3s &,
               const ContactMap &, const int>(
          bp::args("self", "space", "total mass", "gravity", "contact_map",
                   "force_size")))
      .def_readwrite("contact_map",
                     &ContinuousCentroidalFwdDynamics::contact_map_)
      .def(CreateDataPythonVisitor<ContinuousCentroidalFwdDynamics>())
      .def(ode_visitor);

  bp::register_ptr_to_python<
      shared_ptr<ContinuousCentroidalFwdDataTpl<Scalar>>>();
  bp::class_<ContinuousCentroidalFwdDataTpl<Scalar>, bp::bases<ODEData>>(
      "ContinuousCentroidalFwdData", bp::no_init);
}
} // namespace aligator::python
