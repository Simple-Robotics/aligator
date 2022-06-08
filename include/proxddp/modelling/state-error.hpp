#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function.hpp"
#include <proxnlp/manifold-base.hpp>


namespace proxddp
{

  /// @brief Residual \f$r(x) = x \ominus x_{tar} \f$
  template<typename T>
  struct StateErrorResidual : StageFunctionTpl<T>
  {
    using Scalar = T;
    PROXNLP_DYNAMIC_TYPEDEFS(T);
    using Data = FunctionDataTpl<T>;

    VectorXs target;
    const ManifoldAbstractTpl<T>& space;

    StateErrorResidual(const ManifoldAbstractTpl<T>& space,
                const int nu,
                const VectorXs& target)
      : StageFunctionTpl<T>(space.ndx(), nu, space.ndx())
      , target(target), space(space) {}

    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  Data& data) const
    {
      space.difference(x, target, data.value_);
    }

    void computeJacobians(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& y,
                          Data& data) const
    {
      space.Jdifference(x, target, data.Jx_, 0);
    }
  };
  
} // namespace proxddp

