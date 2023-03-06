/// @file
/// @brief Definitions for Python runtime deprecation warnings.
/// @see Header <pinocchio/bindings/python/utils/deprecation.hpp>
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include <eigenpy/fwd.hpp>

namespace proxddp {
namespace python {
namespace bp = boost::python;

enum class DeprecationTypes { DEPRECATION, PENDING_DEPRECATION, FUTURE };

namespace detail {

constexpr auto deprtype_to_pyobj(DeprecationTypes dep) {
  switch (dep) {
  case DeprecationTypes::DEPRECATION:
    return PyExc_DeprecationWarning;
  case DeprecationTypes::PENDING_DEPRECATION:
    return PyExc_PendingDeprecationWarning;
  case DeprecationTypes::FUTURE:
    return PyExc_FutureWarning;
  }
}

} // namespace detail

/// A Boost.Python policy which triggers a Python warning on precall.
template <class Policy = bp::default_call_policies,
          DeprecationTypes deprecation_type =
              DeprecationTypes::PENDING_DEPRECATION>
struct deprecation_warning_policy : Policy {

  using result_converter = typename Policy::result_converter;
  using argument_package = typename Policy::argument_package;

  deprecation_warning_policy(const std::string &warning_msg = "")
      : Policy(), m_what(warning_msg) {}

  const std::string what() const { return m_what; }

  const Policy *derived() const { return static_cast<const Policy *>(this); }

  template <class ArgPackage> bool precall(const ArgPackage &args) const {
    PyErr_WarnEx(detail::deprtype_to_pyobj(deprecation_type), m_what.c_str(),
                 1);
    return derived()->precall(args);
  }

private:
  const std::string m_what;
};

template <class Policy = bp::default_call_policies,
          DeprecationTypes deprecation_type =
              DeprecationTypes::PENDING_DEPRECATION>
struct deprecated_function
    : deprecation_warning_policy<Policy, deprecation_type> {
  using Base = deprecation_warning_policy<Policy, deprecation_type>;
  deprecated_function(const std::string &warning_msg =
                          "This function has been marked as deprecated, and "
                          "will be removed in the future.")
      : Base(warning_msg) {}
};

template <class Policy = bp::default_call_policies,
          DeprecationTypes deprecation_type =
              DeprecationTypes::PENDING_DEPRECATION>
struct deprecated_member
    : deprecation_warning_policy<Policy, deprecation_type> {
  using Base = deprecation_warning_policy<Policy, deprecation_type>;
  deprecated_member(const std::string &warning_msg =
                        "This attribute has been marked as deprecated, and "
                        "will be removed in the future.")
      : Base(warning_msg) {}
};

} // namespace python
} // namespace proxddp
