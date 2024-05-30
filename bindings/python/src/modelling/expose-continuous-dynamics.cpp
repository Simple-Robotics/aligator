/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include <proxsuite-nlp/python/polymorphic.hpp>
#include "aligator/python/visitors.hpp"
#include "aligator/python/modelling/continuous.hpp"
#include "aligator/modelling/dynamics/centroidal-fwd.hpp"
#include "aligator/modelling/dynamics/continuous-centroidal-fwd.hpp"
#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/contact-map.hpp"
#include "aligator/python/polymorphic-convertible.hpp"

namespace aligator {
namespace python {
using namespace ::aligator::dynamics;
using context::MatrixXs;
using context::Scalar;
using context::VectorXs;

using context::ContinuousDynamicsAbstract;
using context::ContinuousDynamicsData;
using context::ODEAbstract;
using context::ODEData;

using CentroidalFwdDynamics = CentroidalFwdDynamicsTpl<Scalar>;
using ContinuousCentroidalFwdDynamics =
    ContinuousCentroidalFwdDynamicsTpl<Scalar>;
using Vector3s = typename math_types<Scalar>::Vector3s;
using ContactMap = ContactMapTpl<Scalar>;

struct ContinousDataWrapper : ContinuousDynamicsData,
                              bp::wrapper<ContinuousDynamicsData> {
  using ContinuousDynamicsData::ContinuousDynamicsData;
};

void exposeODEs();

void exposeContinuousDynamics() {
  using ManifoldPtr = xyz::polymorphic<context::Manifold>;

  proxsuite::nlp::python::register_polymorphic_to_python<
      xyz::polymorphic<ContinuousDynamicsAbstract>>();
  bp::class_<PyContinuousDynamics<>, boost::noncopyable>(
      "ContinuousDynamicsAbstract",
      "Base class for continuous-time dynamical models (DAEs and ODEs).",
      bp::init<ManifoldPtr, int>("Default constructor: provide the working "
                                 "manifold and control space "
                                 "dimension.",
                                 bp::args("self", "space", "nu")))
      .add_property("ndx", &ContinuousDynamicsAbstract::ndx,
                    "State space dimension.")
      .add_property("nu", &ContinuousDynamicsAbstract::nu,
                    "Control space dimension.")
      .def("evaluate", bp::pure_virtual(&ContinuousDynamicsAbstract::evaluate),
           bp::args("self", "x", "u", "xdot", "data"),
           "Evaluate the DAE functions.")
      .def("computeJacobians",
           bp::pure_virtual(&ContinuousDynamicsAbstract::computeJacobians),
           bp::args("self", "x", "u", "xdot", "data"),
           "Evaluate the DAE function derivatives.")
      .add_property("space",
                    bp::make_function(&ContinuousDynamicsAbstract::space,
                                      bp::return_internal_reference<>()),
                    "Get the state space.")
      .def(CreateDataPolymorphicPythonVisitor<ContinuousDynamicsAbstract,
                                              PyContinuousDynamics<>>());

  bp::register_ptr_to_python<shared_ptr<ContinuousDynamicsData>>();
  auto cont_data_cls =
      bp::class_<ContinousDataWrapper, boost::noncopyable>(
          "ContinuousDynamicsData",
          "Data struct for continuous dynamics/DAE models.",
          bp::init<int, int>(bp::args("self", "ndx", "nu")))
          .add_property("value",
                        bp::make_getter(&ContinuousDynamicsData::value_,
                                        bp::return_internal_reference<>()),
                        "Vector value of the DAE residual.")
          .add_property("Jx",
                        bp::make_getter(&ContinuousDynamicsData::Jx_,
                                        bp::return_internal_reference<>()),
                        "Jacobian with respect to state.")
          .add_property("Ju",
                        bp::make_getter(&ContinuousDynamicsData::Ju_,
                                        bp::return_internal_reference<>()),
                        "Jacobian with respect to controls.")
          .add_property("Jxdot",
                        bp::make_getter(&ContinuousDynamicsData::Jxdot_,
                                        bp::return_internal_reference<>()),
                        "Jacobian with respect to :math:`\\dot{x}`.")
          .add_property("xdot",
                        bp::make_getter(&ContinuousDynamicsData::xdot_,
                                        bp::return_internal_reference<>()),
                        "Time derivative :math:`\\dot{x}`.");

  // Alias this for back-compatibility
  bp::scope().attr("ODEData") = cont_data_cls;

  exposeODEs();
  convertibleToPolymorphicBases<CentroidalFwdDynamics, ODEAbstract,
                                ContinuousDynamicsAbstract>();
  bp::class_<CentroidalFwdDynamics, bp::bases<ODEAbstract>>(
      "CentroidalFwdDynamics",
      "Nonlinear centroidal dynamics with preplanned feet positions",
      bp::init<const proxsuite::nlp::VectorSpaceTpl<Scalar> &, const double,
               const Vector3s &, const ContactMap &, const int>(
          bp::args("self", "space", "total mass", "gravity", "contact_map",
                   "force_size")))
      .def_readwrite("contact_map", &CentroidalFwdDynamics::contact_map_)
      .def(CreateDataPythonVisitor<CentroidalFwdDynamics>());

  bp::register_ptr_to_python<shared_ptr<CentroidalFwdDataTpl<Scalar>>>();
  bp::class_<CentroidalFwdDataTpl<Scalar>, bp::bases<ODEData>>(
      "CentroidalFwdData", bp::no_init);

  convertibleToPolymorphicBases<ContinuousCentroidalFwdDynamics, ODEAbstract,
                                ContinuousDynamicsAbstract>();
  bp::class_<ContinuousCentroidalFwdDynamics, bp::bases<ODEAbstract>>(
      "ContinuousCentroidalFwdDynamics",
      "Nonlinear centroidal dynamics with preplanned feet positions and smooth "
      "forces",
      bp::init<const proxsuite::nlp::VectorSpaceTpl<Scalar> &, const double,
               const Vector3s &, const ContactMap &, const int>(
          bp::args("self", "space", "total mass", "gravity", "contact_map",
                   "force_size")))
      .def_readwrite("contact_map",
                     &ContinuousCentroidalFwdDynamics::contact_map_)
      .def(CreateDataPythonVisitor<ContinuousCentroidalFwdDynamics>());

  bp::register_ptr_to_python<
      shared_ptr<ContinuousCentroidalFwdDataTpl<Scalar>>>();
  bp::class_<ContinuousCentroidalFwdDataTpl<Scalar>, bp::bases<ODEData>>(
      "ContinuousCentroidalFwdData", bp::no_init);
}

} // namespace python
} // namespace aligator
