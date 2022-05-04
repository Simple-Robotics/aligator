#pragma once

#include "proxddp/core/problem.hpp"


namespace proxddp
{
  template<typename _Scalar>
  struct SolverProxDDPTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
  };
  
} // namespace proxddp


