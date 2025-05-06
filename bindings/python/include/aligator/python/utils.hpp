/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include <eigenpy/registration.hpp>
#include <boost/python.hpp>

namespace aligator {
namespace python {
namespace bp = boost::python;

inline std::string get_scope_name(bp::scope scope) {
  return std::string(bp::extract<const char *>(scope.attr("__name__")));
}

/**
 * @brief   Create or retrieve a Python scope (that is, a class or module
 * namespace).
 *
 * @returns The submodule with the input name.
 */
inline bp::object get_namespace(const std::string &name) {
  bp::scope cur_scope; // current scope
  const std::string complete_name = get_scope_name(cur_scope) + "." + name;
  bp::object submodule(bp::borrowed(PyImport_AddModule(complete_name.c_str())));
  cur_scope.attr(name.c_str()) = submodule;
  return submodule;
}

template <typename T> bool register_enum_symlink(bool export_values) {
  namespace bp = boost::python;
  namespace converter = bp::converter;
  if (eigenpy::register_symbolic_link_to_registered_type<T>()) {
    const bp::type_info info = bp::type_id<T>();
    const converter::registration *reg = converter::registry::query(info);
    bp::object class_obj(bp::handle<>(reg->get_class_object()));
    bp::enum_<T> &as_enum = static_cast<bp::enum_<T> &>(class_obj);
    if (export_values)
      as_enum.export_values();
    return true;
  }
  return false;
}

} // namespace python
} // namespace aligator
