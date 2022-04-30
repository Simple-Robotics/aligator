#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/node-function.hpp"


namespace proxddp
{
  /** @brief Stage costs \f$ \ell(x, u) \f$ for control problems.
   */
  template<typename _Scalar>
  struct StageCostTpl : StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
  };
  
} // namespace proxddp

