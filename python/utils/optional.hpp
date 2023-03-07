/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include <boost/optional.hpp>
#include <eigenpy/fwd.hpp>
#include <eigenpy/eigen-from-python.hpp>

namespace boost {
namespace python {
namespace converter {

template <typename T>
struct expected_pytype_for_arg<boost::optional<T>>
    : expected_pytype_for_arg<T> {};

} // namespace converter
} // namespace python
} // namespace boost

namespace proxddp {
namespace python {
namespace bp = boost::python;

namespace detail {

template <typename T> struct OptionalToPython {
  static PyObject *convert(const boost::optional<T> &obj) {
    if (obj)
      return bp::incref(bp::object(*obj).ptr());
    else {
      return bp::incref(bp::object().ptr()); // None
    }
  }

  static PyTypeObject const *get_pytype() {
    return bp::converter::registered_pytype<T>::get_pytype();
  }

  static void registration() {
    bp::to_python_converter<boost::optional<T>, OptionalToPython, true>();
  }
};

template <typename T> struct OptionalFromPython {
  static void *convertible(PyObject *obj_ptr);

  static void construct(PyObject *obj_ptr,
                        bp::converter::rvalue_from_python_stage1_data *memory);

  static void registration();
};

template <typename T>
void *OptionalFromPython<T>::convertible(PyObject *obj_ptr) {
  if (obj_ptr == Py_None) {
    return obj_ptr;
  }
  bp::extract<T> bp_obj(obj_ptr);
  if (!bp_obj.check())
    return 0;
  else
    return obj_ptr;
}

template <typename T>
void OptionalFromPython<T>::construct(
    PyObject *obj_ptr, bp::converter::rvalue_from_python_stage1_data *memory) {
  // create storage
  using rvalue_storage_t =
      bp::converter::rvalue_from_python_storage<boost::optional<T>>;
  void *storage =
      reinterpret_cast<rvalue_storage_t *>(reinterpret_cast<void *>(memory))
          ->storage.bytes;

  if (obj_ptr == Py_None) {
    new (storage) boost::optional<T>(boost::none);
  } else {
    const T value = bp::extract<T>(obj_ptr);
    new (storage) boost::optional<T>(value);
  }

  memory->convertible = storage;
}

template <typename T> void OptionalFromPython<T>::registration() {
  bp::converter::registry::push_back(&convertible, &construct,
                                     bp::type_id<boost::optional<T>>()
#if false
      ,
      bp::converter::expected_pytype_for_arg<boost::optional<T>>::get_pytype
#endif
  );
}

} // namespace detail

/// Register converters for the type `boost::optional<T>` to Python.
template <typename T> struct OptionalConverter {
  static void registration() {
    detail::OptionalToPython<T>::registration();
    detail::OptionalFromPython<T>::registration();
  }
};

} // namespace python
} // namespace proxddp
