/// @file macros.hpp
/// @brief Macros for Boost.Python, inspired by Pybind11's macros.
#pragma once

#include <type_traits>
#include <fmt/format.h>
#include <stdexcept>


namespace proxddp { namespace python {
  namespace internal
  {
    template<bool B, class T = void>
    using enable_if_t = typename std::enable_if<B,T>::type;

    /// Template function enabled when template arg ret_type is void.
    /// In this case, suppress the return.
    template<typename ret_type, typename T>
    enable_if_t<std::is_same<ret_type, void>::value> suppress_if_void(T&&) {}

    template<typename T> T suppress_if_void(T&& o) { return std::forward<T>(o); }
    
  } // namespace internal
} // namespace python
} // namespace proxddp


#define PROXDDP_PYTHON_OVERRIDE_IMPL(ret_type, pyname, ...)                          \
  do {                                                                               \
    if (bp::override fo = this->get_override(pyname))                                \
    {                                                                                \
      auto o = fo(__VA_ARGS__);                                                      \
      return ::proxddp::python::internal::suppress_if_void<ret_type>(std::move(o));  \
    }                                                                                \
  } while(false)

/** 
 * @def PROXDDP_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)
 * @brief Define the body of a virtual function override. This is meant
 *        to reduce boilerplate code when exposing virtual member functions.
 */
#define PROXDDP_PYTHON_OVERRIDE_PURE(ret_type, pyname, ...)                                   \
  PROXDDP_PYTHON_OVERRIDE_IMPL(ret_type, pyname, __VA_ARGS__);                                \
  throw std::runtime_error(fmt::format("Tried to call pure virtual function {:s}.", pyname))

/**
 * @def PROXDDP_PYTHON_OVERRIDE(ret_type, cname, fname, ...)
 * @copybrief PROXDDP_PYTHON_OVERRIDE_PURE()
 */
#define PROXDDP_PYTHON_OVERRIDE(ret_type, cname, fname, ...)                         \
  PROXDDP_PYTHON_OVERRIDE_IMPL(ret_type, #fname, __VA_ARGS__);                       \
  return cname::fname(__VA_ARGS__)
