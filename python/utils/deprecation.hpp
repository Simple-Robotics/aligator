/// @file
/// @brief Definitions for Python runtime deprecation warnings.
/// @see Header <pinocchio/bindings/python/utils/deprecation.hpp>
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include <eigenpy/fwd.hpp>

namespace proxddp {
namespace python {
namespace bp = boost::python;

/// A Boost.Python policy which triggers a Python warning on precall.
template <class Policy = bp::default_call_policies>
struct deprecation_warning_policy : Policy {

  using result_converter = typename Policy::result_converter;
  using argument_package = typename Policy::argument_package;

  deprecation_warning_policy(const std::string &warning_msg = "")
      : Policy(), m_what(warning_msg) {}

  const std::string what() const { return m_what; }

  const Policy *derived() const { return static_cast<const Policy *>(this); }

  template <class ArgPackage> bool precall(const ArgPackage &args) const {
    PyErr_WarnEx(PyExc_UserWarning, m_what.c_str(), 1);
    return derived()->precall(args);
  }

private:
  const std::string m_what;
};

template <class Policy = bp::default_call_policies>
struct deprecated_member : deprecation_warning_policy<Policy> {
  using Base = deprecation_warning_policy<Policy>;
  deprecated_member(const std::string &warning_msg =
                        "This attribute has been marked as deprecated, and "
                        "will be removed in the future.")
      : Base(warning_msg) {}
};

} // namespace python
} // namespace proxddp
