/// @file macros.hpp
/// @brief Macros for Boost.Python, inspired by Pybind11's macros.
#pragma once

#include <type_traits>


namespace proxddp { namespace python {
  namespace internal
  {
    /// Template function enabled when template arg ret_type is void.
    /// In this case, suppress the return.
    /// Used in 
    template<typename ret_type, typename T>
    typename std::enable_if<std::is_same<ret_type, void>::value, void>::type
    suppress_if_void(T&&) {}

    template<typename T> T suppress_if_void(T&& o) { return std::move(o); }
    
  } // namespace internal
} // namespace python
} // namespace proxddp


#define PROXDDP_PYTHON_OVERRIDE_IMPL(ret_type, pyname, ...)  \
  bp::override fo = get_override(pyname);           \
  if (fo)                                           \
  {                                                 \
    return ::proxddp::python::internal::suppress_if_void<ret_type>(fo(__VA_ARGS__));}  \

/** @def PROXDDP_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)
 * @brief Define the body of a virtual function override. This is meant
 *        to reduce boilerplate code when exposing virtual member functions.
 */
#define PROXDDP_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)      \
  PROXDDP_PYTHON_OVERRIDE_IMPL(ret_type, pyname, __VA_ARGS__)


/**
 * @def
 * @brief Define the body of a non-pure virtual function override.
 */
#define PROXDDP_PYTHON_OVERRIDE(ret_type, cname, fname, ...)  \
  PROXDDP_PYTHON_OVERRIDE_IMPL(ret_type, #fname, __VA_ARGS__) \
  return cname::fname(__VA_ARGS__)
