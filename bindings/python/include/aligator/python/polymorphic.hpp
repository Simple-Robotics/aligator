#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

#include <boost/core/demangle.hpp>
#include <variant>

// Required class template specialization for
// boost::python::register_ptr_to_python<> to work.
namespace boost::python {
template <class T, class A> struct pointee<xyz::polymorphic<T, A>> {
  typedef T type;
};
} // namespace boost::python

namespace aligator {
namespace python {
namespace bp = boost::python;

/// @brief Expose a polymorphic value type, e.g. xyz::polymorphic<T, A>.
/// @details Just an alias for bp::register_ptr_to_python<>().
template <class Poly> inline void register_polymorphic_to_python() {
  using X = typename bp::pointee<Poly>::type;
  bp::objects::class_value_wrapper<
      Poly, bp::objects::make_ptr_instance<
                X, bp::objects::pointer_holder<Poly, X>>>();
}

template <class Poly> struct PolymorphicVisitor;

/// Does the same thing as boost::python::implicitly_convertible<>(),
/// except the conversion is placed at the top of the conversion chain.
/// This ensures that Boost.Python attempts to convert to Poly BEFORE
/// any parent class!
///
/// Example:
///
///   struct X {
///     virtual ~X() = default;
///     virtual std::string name() const { return "X"; }
///   };
///   struct PyX final : X, aligator::python::PolymorphicWrapper<PyX, X> {
///     std::string name() const override {
///       if (boost::python::override f = get_override("name")) {
///         return f();
///       }
///       return X::name();
///     }
///   };
///   using PolyX = xyz::polymorphic<X>;
///
///   namespace boost::python::objects {
///
///   template <>
///   struct value_holder<PyX> :
///   aligator::python::OwningNonOwningHolder<PyX>
///   {
///     using OwningNonOwningHolder::OwningNonOwningHolder;
///   };
///
///   } // namespace boost::python::objects
///
///   BOOST_PYTHON_MODULE(module_name) {
///     boost::python::class_<PyX, boost::noncopyable>("X",
///     boost::python::init<>())
///         .def("name", &X::name)
///         .def(aligator::python::PolymorphicVisitor<PolyX>());
///   }

template <class Base, class A>
struct PolymorphicVisitor<xyz::polymorphic<Base, A>>
    : bp::def_visitor<PolymorphicVisitor<xyz::polymorphic<Base, A>>> {
  using Poly = xyz::polymorphic<Base, A>;
  static_assert(std::is_polymorphic_v<Base>, "Type should be polymorphic!");

  template <class PyClass> void visit(PyClass &cl) const {
    using T = typename PyClass::wrapped_type;
    using meta = typename PyClass::metadata;
    using held = typename meta::held_type;
    typedef bp::converter::implicit<held, Poly> functions;

    ALIGATOR_COMPILER_DIAGNOSTIC_PUSH
    ALIGATOR_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
    bp::converter::registry::insert(
        &functions::convertible, &functions::construct, bp::type_id<Poly>(),
        &bp::converter::expected_from_python_type_direct<T>::get_pytype);
    ALIGATOR_COMPILER_DIAGNOSTIC_POP

    // Enable pickling of the Derived Python class.
    // If a Python class inherit from cl, this call will allow pickle to
    // serialize/unserialize all the Python class content.
    // The C++ part will not be serialized but it's managed differently
    // by PolymorphicWrapper.
    if constexpr (std::is_base_of_v<boost::python::wrapper<Base>,
                                    typename PyClass::wrapped_type>) {
      cl.enable_pickling_(true);
    }
  }
};

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

// Specialize value_holder to allow switching to a non owning holder.
// This code is similar to boost::python::value_holder code but allow to switch
// between value or ptr at runtime.
template <typename Value>
struct OwningNonOwningHolder : boost::python::instance_holder {
  typedef Value held_type;
  typedef Value value_type;

private:
  struct PtrGetter {
    Value *operator()(Value &v) { return &v; };
    Value *operator()(Value *v) { return v; };
  };

  Value *get_ptr() { return std::visit(PtrGetter(), m_held); }

public:
  template <typename... Args>
  OwningNonOwningHolder(PyObject *self, Args... args)
      : m_held(Value(boost::python::objects::do_unforward(
            std::forward<Args>(args), 0)...)) {
    boost::python::detail::initialize_wrapper(self, get_ptr());
  }

private: // required holder implementation
  void *holds(boost::python::type_info dst_t, bool) {
    if (void *wrapped = holds_wrapped(dst_t, get_ptr(), get_ptr()))
      return wrapped;

    boost::python::type_info src_t = boost::python::type_id<Value>();
    return src_t == dst_t ? get_ptr()
                          : boost::python::objects::find_static_type(
                                get_ptr(), src_t, dst_t);
  }

  template <class T>
  inline void *holds_wrapped(boost::python::type_info dst_t,
                             boost::python::wrapper<T> *, T *p) {
    return boost::python::type_id<T>() == dst_t ? p : 0;
  }

  inline void *holds_wrapped(boost::python::type_info, ...) { return 0; }

public: // data members
  std::variant<Value, Value *> m_held;
};

// This class replace boost::python::wrapper when we want to use
// xyz::polymorphic to hold polymorphic class.
//
// The following class diagram describe how boost::python::wrapper work:
//
// PyDerived -----------|
//    ^      1  m_self  |
//    |                 |
//  PyBase <--- wrapper--
//    ^
//    |
//   Base
//
// PyDerived is a Python class that inherit from PyBase.
// Wrapper will hold a non owning ptr to PyDerived.
// This can lead to ownership issue when PyBase is stored in C++ object.
//
// To avoid this issue, at each PolymorphicWrapper copy, we will deep copy and
// own m_self in copied_owner.
// When copied, m self will default construct a new PyBase instance.
// To avoid having two PyBase instance (the one copied in C++ and the other one
// default constructed by Boost.Python) we store PyBase in a custom holder
// (OwningNonOwningHolder) and we reset this holder to hold a ptr to the new C++
// instance.
template <typename _PyBase, typename _Base>
struct PolymorphicWrapper : boost::python::wrapper<_Base> {
  using PyBase = _PyBase;
  using Base = _Base;

  PolymorphicWrapper() = default;
  PolymorphicWrapper(const PolymorphicWrapper &o) : bp::wrapper<Base>(o) {
    deepcopy_owner();
  }
  PolymorphicWrapper(PolymorphicWrapper &&o) = default;

  PolymorphicWrapper &operator=(const PolymorphicWrapper &o) {
    if (this == &o) {
      return *this;
    }
    bp::wrapper<Base>::operator=(o);
    deepcopy_owner();
    return *this;
  }
  PolymorphicWrapper &operator=(PolymorphicWrapper &&o) = default;

private:
  void deepcopy_owner() {
    namespace bp = boost::python;
    if (PyObject *owner_ptr = bp::detail::wrapper_base_::get_owner(*this)) {
      bp::object copy = bp::import("copy");
      bp::object deepcopy = copy.attr("deepcopy");

      // bp::object decrements the refcount at destruction
      // so by calling it with borrowed we incref owner_ptr to avoid destroying
      // it.
      bp::object owner{bp::handle<>(bp::borrowed(owner_ptr))};

      // Deepcopy is safer to copy derived class values but don't copy C++ class
      // content.
      // We store the copied owner into a member variable to take the
      // ownership.
      copied_owner = deepcopy(owner);
      PyObject *copied_owner_ptr = copied_owner.ptr();

      // copied_owner contains a OwningNonOwningHolder<PyBase> that contains a
      // PyBase (different from this).
      // We modify this value_holder specialization to store this ptr (non
      // owning) instead. This will avoid that Python and C++ work with
      // different object. Since copied_owner is owned by this, there is no
      // memory leak.
      // TODO check value holder type
      bp::objects::instance<> *inst =
          ((bp::objects::instance<> *)copied_owner_ptr);
      OwningNonOwningHolder<PyBase> *value_holder =
          dynamic_cast<OwningNonOwningHolder<PyBase> *>(inst->objects);
      if (!value_holder) {
        std::ostringstream error_msg;
        error_msg << "OwningNonOwningHolder should be setup for "
                  << boost::core::demangle(typeid(PyBase).name()) << " type"
                  << std::endl;
        throw std::logic_error(error_msg.str());
      }
      value_holder->m_held = static_cast<PyBase *>(this);

      bp::detail::initialize_wrapper(copied_owner_ptr, this);
    }
  }

private:
  bp::object copied_owner;
};

namespace internal {

/// Use the same trick from <eigenpy/eigen-to-python.hpp> to specialize the
/// template for both const and non-const.
/// This specialization do the same thing than to_python_indirect with the
/// following differences:
/// - Return the content of the xyz::polymorphic
template <class poly_ref, class MakeHolder> struct ToPythonIndirectPoly {
  using poly_type = boost::remove_cv_ref_t<poly_ref>;

  template <class U> PyObject *operator()(U const &x) const {
    return execute(const_cast<U &>(x));
  }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
  PyTypeObject const *get_pytype() {
    return boost::python::converter::registered_pytype<poly_type>::get_pytype();
  }
#endif

private:
  template <class T, class A>
  static PyObject *execute(const xyz::polymorphic<T, A> &p) {
    if (p.valueless_after_move())
      return bp::detail::none();
    T *q = const_cast<T *>(boost::get_pointer(p));
    assert(q);
    return bp::to_python_indirect<const T &, MakeHolder>{}(q);
  }
};

} // namespace internal
} // namespace python
} // namespace aligator

namespace boost {
namespace python {

template <class T, class A, class MakeHolder>
struct to_python_indirect<xyz::polymorphic<T, A> &, MakeHolder>
    : aligator::python::internal::ToPythonIndirectPoly<xyz::polymorphic<T, A> &,
                                                       MakeHolder> {};

template <class T, class A, class MakeHolder>
struct to_python_indirect<const xyz::polymorphic<T, A> &, MakeHolder>
    : aligator::python::internal::ToPythonIndirectPoly<
          const xyz::polymorphic<T, A> &, MakeHolder> {};

} // namespace python
} // namespace boost
