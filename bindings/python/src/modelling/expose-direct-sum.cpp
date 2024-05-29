/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#include <proxsuite-nlp/python/polymorphic.hpp>
#include "aligator/context.hpp"
#include "aligator/python/fwd.hpp"
#include "aligator/modelling/explicit-dynamics-direct-sum.hpp"

namespace aligator {
namespace python {

using context::Scalar;
using DirectSumExplicitDynamics = DirectSumExplicitDynamicsTpl<Scalar>;
using context::DynamicsModel;
using context::ExplicitDynamics;

void exposeExplicitDynDirectSum() {

  bp::implicitly_convertible<DirectSumExplicitDynamics,
                             xyz::polymorphic<ExplicitDynamics>>();
  bp::implicitly_convertible<DirectSumExplicitDynamics,
                             xyz::polymorphic<DynamicsModel>>();
  proxsuite::nlp::python::register_polymorphic_to_python<
      xyz::polymorphic<DirectSumExplicitDynamics>>();
  bp::class_<DirectSumExplicitDynamics, bp::bases<ExplicitDynamics>>(
      "DirectSumExplicitDynamics",
      "Direct sum :math:`f \\oplus g` of two explicit dynamical models.",
      bp::no_init)
      .def(bp::init<xyz::polymorphic<ExplicitDynamics>,
                    xyz::polymorphic<ExplicitDynamics>>(
          bp::args("self", "f", "g")));

  bp::class_<DirectSumExplicitDynamics::Data,
             bp::bases<context::ExplicitDynamicsData>>(
      "DirectSumExplicitDynamicsData", bp::no_init)
      .def_readwrite("data1", &DirectSumExplicitDynamics::Data::data1_)
      .def_readwrite("data2", &DirectSumExplicitDynamics::Data::data2_);

  bp::def("directSum", directSum<Scalar>, bp::args("f", "g"),
          "Produce the direct sum.");
}

} // namespace python
} // namespace aligator
