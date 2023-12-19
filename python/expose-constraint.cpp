#include "proxddp/python/fwd.hpp"

#include "proxddp/core/constraint.hpp"
#include <proxsuite-nlp/constraint-base.hpp>

namespace proxddp {
namespace python {

context::StageConstraint *
make_constraint_wrap(const shared_ptr<context::StageFunction> &f,
                     const shared_ptr<context::ConstraintSet> &c) {
  return new context::StageConstraint{f, c};
}

void exposeConstraint() {
  using context::ConstraintSet;
  using context::ConstraintStack;
  using context::StageConstraint;

  bp::class_<StageConstraint>(
      "StageConstraint",
      "A stage-wise constraint, of the form :math:`c(x,u) \\leq 0 c(x,u)`.\n"
      ":param f: underlying function\n"
      ":param cs: constraint set",
      bp::no_init)
      .def("__init__",
           bp::make_constructor(make_constraint_wrap,
                                bp::default_call_policies(),
                                bp::args("func", "cstr_set")),
           "Contruct a StageConstraint from a StageFunction and a constraint "
           "set.")
      .def_readwrite("func", &StageConstraint::func)
      .def_readwrite("set", &StageConstraint::set)
      .add_property("nr", &StageConstraint::nr, "Get constraint dimension.");

  bp::class_<ConstraintStack>("ConstraintStack", "The stack of constraint.",
                              bp::no_init)
      .add_property("size", &ConstraintStack::size,
                    "Get number of individual constraints.")
      .add_property("dims",
                    bp::make_function(&ConstraintStack::getDims,
                                      bp::return_internal_reference<>()),
                    "Get the individual dimensions of all constraints.")
      .def(eigenpy::details::overload_base_get_item_for_std_vector<
           ConstraintStack>())
      .add_property("total_dim", &ConstraintStack::totalDim,
                    "Get total dimension of all constraints.");
}

} // namespace python
} // namespace proxddp
