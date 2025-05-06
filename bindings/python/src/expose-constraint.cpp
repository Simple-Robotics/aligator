#include "aligator/python/fwd.hpp"

#include "aligator/core/constraint.hpp"
#include "aligator/core/constraint-set.hpp"

#include <eigenpy/deprecation-policy.hpp>

namespace aligator::python {
using context::ConstraintSet;
using context::ConstraintStack;
using context::StageConstraint;
using context::StageFunction;
using PolyFunc = xyz::polymorphic<StageFunction>;
using PolySet = xyz::polymorphic<ConstraintSet>;

context::StageConstraint *make_constraint_wrap(const PolyFunc &f,
                                               const PolySet &c) {
  return new context::StageConstraint{f, c};
}

void exposeConstraintSets();

void exposeConstraint() {
  exposeConstraintSets();

  bp::class_<StageConstraint>(
      "StageConstraint",
      "A stage-wise constraint, of the form :math:`c(x,u) \\leq 0 c(x,u)`.\n"
      ":param f: underlying function\n"
      ":param cs: constraint set",
      bp::no_init)
      .def("__init__",
           bp::make_constructor(
               make_constraint_wrap,
               eigenpy::deprecation_warning_policy<>(
                   "This class has been deprecated and will be removed in a "
                   "future version of aligator."),
               ("func"_a, "cstr_set")),
           "Contruct a StageConstraint from a StageFunction and a constraint "
           "set.")
      .def_readwrite("func", &StageConstraint::func)
      .def_readwrite("set", &StageConstraint::set)
      .add_property(
          "nr", +[](StageConstraint const &el) { return el.func->nr; },
          "Get constraint dimension.");

  bp::class_<ConstraintStack>("ConstraintStack", "The stack of constraint.",
                              bp::no_init)
      .add_property("size", &ConstraintStack::size,
                    "Get number of individual constraints.")
      .def_readonly("funcs", &ConstraintStack::funcs)
      .def_readonly("sets", &ConstraintStack::sets)
      .add_property("dims",
                    bp::make_function(&ConstraintStack::dims,
                                      bp::return_internal_reference<>()),
                    "Get the individual dimensions of all constraints.")
      .add_property("total_dim", &ConstraintStack::totalDim,
                    "Get total dimension of all constraints.");
}

} // namespace aligator::python
