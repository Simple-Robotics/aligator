#include "proxddp/python/fwd.hpp"

#include "proxddp/core/constraint.hpp"
#include <proxnlp/constraint-base.hpp>

namespace proxddp {
namespace python {

context::StageConstraint *
make_constraint_wrap(const shared_ptr<context::StageFunction> &f,
                     const shared_ptr<context::ConstraintSet> &c) {
  return new context::StageConstraint{f, c};
}

void exposeConstraint() {
  using context::ConstraintSet;
  using context::StageConstraint;

  bp::class_<context::StageConstraint>(
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
      .def_readwrite("func", &context::StageConstraint::func)
      .def_readwrite("set", &context::StageConstraint::set);

  bp::class_<context::ConstraintStack>("ConstraintStack",
                                       "The stack of constraint.", bp::no_init);
}

} // namespace python
} // namespace proxddp
