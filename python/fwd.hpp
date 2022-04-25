#pragma once

#include "proxddp/python/context.hpp"

#include <eigenpy/eigenpy.hpp>

namespace proxddp
{
namespace python
{
  namespace bp = boost::python;
  
  void exposeNode();

} // namespace python
} // namespace proxddp

