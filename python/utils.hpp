/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once
#include <proxnlp/python/utils/namespace.hpp>
#include <eigenpy/registration.hpp>

namespace proxddp {
namespace python {

using proxnlp::python::get_namespace;
using proxnlp::python::get_scope_name;

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
} // namespace proxddp
