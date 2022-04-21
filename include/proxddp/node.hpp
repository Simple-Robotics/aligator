#pragma once

#include "proxnlp/function-base.hpp"
#include "proxnlp/cost-function.hpp"

#include "proxnlp/manifold-base.hpp"


namespace proxddp
{

  /** @brief    A node in the control problem.
   */
  template<typename _Scalar>
  struct NodeTpl
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTOR_TYPEDEFS(Scalar)

    using Manifold = ManifoldAbstractTpl<Scalar>;

    std::unique_ptr<Manifold> man;

  };

} // namespace proxddp

