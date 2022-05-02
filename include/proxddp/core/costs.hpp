#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function.hpp"


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

