#pragma once

#include "aligator/python/fwd.hpp"
#include <proxsuite-nlp/python/polymorphic.hpp>

#include <boost/mpl/vector.hpp>

namespace aligator::python {
using proxsuite::nlp::python::PolymorphicVisitor;
using proxsuite::nlp::python::register_polymorphic_to_python;

/// Declare concrete to be implicitly convertible to
/// polymorphic<U> for a set of base classes @tparam Bases passed as the
/// variadic template arguments.
template <class... Bases>
struct PolymorphicMultiBaseVisitor
    : bp::def_visitor<PolymorphicMultiBaseVisitor<Bases...>> {
  // ptr to avoid instantiating abstract class
  using types = boost::mpl::vector<Bases *...>;

  template <class... Args> void visit(bp::class_<Args...> &cl) const {
    // use the lambda to expose conversion to polymorphic<base> for each
    // of the types in the variadic parameter Bases
    boost::mpl::for_each<types>([&cl](auto *arg = 0) {
      // get the type of the passed argument
      using base_t = std::remove_reference_t<decltype(*arg)>;
      cl.def(PolymorphicVisitor<xyz::polymorphic<base_t>>{});
    });
  }
};

} // namespace aligator::python
