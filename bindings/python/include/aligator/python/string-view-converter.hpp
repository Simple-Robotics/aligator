#pragma once

#include <boost/python.hpp>

namespace bp = boost::python;

namespace aligator::python {
namespace detail {

struct string_view_to_python {
  static PyObject *convert(std::string_view sv) noexcept {
    return PyUnicode_FromStringAndSize(sv.data(),
                                       static_cast<Py_ssize_t>(sv.size()));
  }
};

struct string_view_from_python {
  static void *convertible(PyObject *obj_ptr) noexcept {
    if (!PyUnicode_Check(obj_ptr) && !PyBytes_Check(obj_ptr)) {
      return nullptr;
    }
    return obj_ptr;
  }

  static void construct(PyObject *obj_ptr,
                        bp::converter::rvalue_from_python_stage1_data *data) {
    const char *value = nullptr;
    Py_ssize_t len = 0;

    if (PyUnicode_Check(obj_ptr)) {
      value = PyUnicode_AsUTF8AndSize(obj_ptr, &len);
      if (!value) {
        bp::throw_error_already_set();
      }

      void *storage =
          ((bp::converter::rvalue_from_python_storage<std::string_view> *)data)
              ->storage.bytes;
      new (storage) std::string_view(value, static_cast<size_t>(len));
      data->convertible = storage;
    }
  }
};
} // namespace detail

inline void register_string_view_converter() {
  bp::to_python_converter<std::string_view, detail::string_view_to_python>();

  bp::converter::registry::push_back(
      &detail::string_view_from_python::convertible,
      &detail::string_view_from_python::construct,
      bp::type_id<std::string_view>());
}

} // namespace aligator::python
