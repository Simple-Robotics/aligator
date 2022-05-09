///   Defines the constraint object for this library.
#pragma once

#include "proxddp/core/function.hpp"


namespace proxddp
{

  /** @brief  Base class for stage-wise constraint objects.
   * 
   * This class packs a StageFunctionTpl and ConstraintSetBase together.
   * It models stage-wise constraints of the form
   * \f[
   *        c(x, u, x') \in \mathcal{C}.
   * \f]
   */
  template<typename _Scalar>
  struct StageConstraintTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

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

