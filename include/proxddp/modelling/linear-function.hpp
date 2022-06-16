#pragma once

#include "proxddp/core/function.hpp"


namespace proxddp
{
  template<typename Scalar>
  struct LinearFunction : StageFunctionTpl<Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  };
  
} // namespace proxddp

