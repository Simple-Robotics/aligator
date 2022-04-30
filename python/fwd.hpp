#pragma once


namespace proxddp
{
  /// @brief  The Python bindings.
  namespace python {}
} // namespace proxddp

#include "proxddp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>

namespace proxddp
{
  namespace python
  {
    namespace bp = boost::python;
    
    /// Expose ternary functions
    void exposeFunctions();
    void exposeNode();
    void exposeIntegrators();

  } // namespace python
} // namespace proxddp

