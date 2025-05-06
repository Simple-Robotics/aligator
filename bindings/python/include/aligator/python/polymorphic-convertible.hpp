#pragma once

#include "aligator/python/fwd.hpp"

namespace aligator::python {
using proxsuite::nlp::python::PolymorphicVisitor;
using proxsuite::nlp::python::register_polymorphic_to_python;

/// Declare to Boost.Python that a given class is implicitly convertible to
/// polymorphic<U> for a set of base classes @tparam Bases passed as the
/// variadic template arguments.
template <class... Bases>
struct PolymorphicMultiBaseVisitor
    : bp::def_visitor<PolymorphicMultiBaseVisitor<Bases...>> {

  template <class... Args> void visit(bp::class_<Args...> &cl) const {
    (cl.def(PolymorphicVisitor<xyz::polymorphic<Bases>>{}), ...);
  }
};

} // namespace aligator::python
