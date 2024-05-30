#include "aligator/python/fwd.hpp"

#include <boost/mpl/vector.hpp>

namespace aligator::python {

namespace detail {
template <class T> struct ConvertiblePolymorphicBases {
  template <class U> void operator()(U *) {
    bp::implicitly_convertible<T, xyz::polymorphic<U>>();
  }
};
} // namespace detail

/// Declare concrete type @tparam T to be implicitly convertible to
/// polymorphic<U> for a set of base classes U passed as the variadic template
/// arguments.
template <class T, class... Bases> void convertibleToPolymorphicBases() {
  using types = boost::mpl::vector<Bases *...>;
  boost::mpl::for_each<types>(detail::ConvertiblePolymorphicBases<T>{});
}

} // namespace aligator::python
