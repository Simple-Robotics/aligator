/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#include "aligator/context.hpp"
#include "aligator/python/fwd.hpp"
#include "aligator/modelling/explicit-dynamics-direct-sum.hpp"

namespace aligator {
namespace python {

using context::Scalar;
using DirectSumExplicitDynamics = DirectSumExplicitDynamicsTpl<Scalar>;
using context::ExplicitDynamics;

void exposeExplicitDynDirectSum() {
  PolymorphicMultiBaseVisitor<ExplicitDynamics> exp_dynamics_visitor;

  register_polymorphic_to_python<xyz::polymorphic<DirectSumExplicitDynamics>>();
  bp::class_<DirectSumExplicitDynamics, bp::bases<ExplicitDynamics>>(
      "DirectSumExplicitDynamics",
      "Direct sum :math:`f \\oplus g` of two explicit dynamical models.",
      bp::no_init)
      .def(bp::init<xyz::polymorphic<ExplicitDynamics>,
                    xyz::polymorphic<ExplicitDynamics>>(("self"_a, "f", "g")))
      .def(exp_dynamics_visitor);

  bp::class_<DirectSumExplicitDynamics::Data,
             bp::bases<context::ExplicitDynamicsData>>(
      "DirectSumExplicitDynamicsData", bp::no_init)
      .def_readwrite("data1", &DirectSumExplicitDynamics::Data::data1_)
      .def_readwrite("data2", &DirectSumExplicitDynamics::Data::data2_);

  bp::def("directSum", directSum<Scalar>, ("f"_a, "g"),
          "Produce the direct sum.");
}

} // namespace python
} // namespace aligator
