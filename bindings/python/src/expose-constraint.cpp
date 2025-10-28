#include "aligator/python/fwd.hpp"
#include "aligator/python/utils.hpp"

#include "aligator/core/constraint.hpp"
#include "aligator/core/constraint-set.hpp"

#include <eigenpy/deprecation-policy.hpp>

namespace aligator::python {
using context::ConstraintSet;
using context::ConstraintStack;
using context::StageFunction;
using PolyFunc = xyz::polymorphic<StageFunction>;
using PolySet = xyz::polymorphic<ConstraintSet>;

void exposeConstraintSets();

void exposeConstraint() {
  {
    bp::scope scope = get_namespace("constraints");
    exposeConstraintSets();
  }

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
