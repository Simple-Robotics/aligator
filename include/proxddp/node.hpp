#pragma once

#include "proxddp/fwd.hpp"

#include "proxnlp/function-base.hpp"
#include "proxnlp/cost-function.hpp"
#include "proxnlp/manifold-base.hpp"

#include <memory>


namespace proxddp
{

  /** @brief    A node in the control problem.
   */
  template<typename _Scalar>
  struct StageModelTpl
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTOR_TYPEDEFS(Scalar)

    using Manifold = ManifoldAbstractTpl<Scalar>;

    shared_ptr<Manifold> space;

    StageModelTpl(const shared_ptr<Manifold>& space)
      : space(space)
        {}
    

  };

} // namespace proxddp

