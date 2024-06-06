#include "aligator/python/fwd.hpp"

#include "aligator/core/constraint.hpp"
#include <proxsuite-nlp/constraint-base.hpp>

namespace aligator {
namespace python {

void exposeConstraint() {
  using context::ConstraintSet;
  using context::ConstraintStack;
  using context::StageConstraint;
  using context::StageFunction;
  using PolyFunc = xyz::polymorphic<StageFunction>;
  using PolySet = xyz::polymorphic<ConstraintSet>;

  bp::class_<StageConstraint>(
      "StageConstraint",
      "A stage-wise constraint, of the form :math:`c(x,u) \\leq 0 c(x,u)`.\n"
      ":param f: underlying function\n"
      ":param cs: constraint set",
      bp::no_init)
      .def(bp::init<const PolyFunc &, const PolySet &>(
          ("self"_a, "func", "cstr_set"),
          "Contruct a StageConstraint from a StageFunction and a constraint "
          "set."))
      .def_readwrite("func", &StageConstraint::func)
      .def_readwrite("set", &StageConstraint::set)
      .add_property("nr", &StageConstraint::nr, "Get constraint dimension.");

  bp::class_<ConstraintStack>("ConstraintStack", "The stack of constraint.",
                              bp::no_init)
      .add_property("size", &ConstraintStack::size,
                    "Get number of individual constraints.")
      .add_property("dims",
                    bp::make_function(&ConstraintStack::dims,
                                      bp::return_internal_reference<>()),
                    "Get the individual dimensions of all constraints.")
      .def(eigenpy::details::overload_base_get_item_for_std_vector<
           ConstraintStack>())
      .add_property("total_dim", &ConstraintStack::totalDim,
                    "Get total dimension of all constraints.");
}

} // namespace python
} // namespace aligator
