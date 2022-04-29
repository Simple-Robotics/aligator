/// @file   Defines the constraint object for this library.
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/node-function.hpp"


namespace proxddp
{

  template<typename _Scalar>
  struct StageConstraintTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

    using FunctionType = StageFunctionTpl<Scalar>;
    using ConstraintSet = ConstraintSetBase<Scalar>;
    using ConstraintSetPtr = shared_ptr<ConstraintSetBase<Scalar>>;

    const FunctionType& func_;
    const ConstraintSet& set_;

    StageConstraintTpl(const FunctionType& func,
                       const ConstraintSet& constraint_set)
      : func_(func)
      , set_(constraint_set)
    {}

    ConstraintSetBase<Scalar>& getConstraintSet() const
    {
      return *set_;
    }

  };
  
} // namespace proxddp

