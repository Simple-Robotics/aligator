#pragma once

#include "proxddp/fwd.hpp"


namespace proxddp
{
  namespace python
  {

    namespace context
    {
      using Scalar = double;

      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

      using Manifold = ManifoldAbstractTpl<Scalar>;
      using Constraint = ConstraintSetBase<Scalar>;
      
    } // namespace context

  } // namespace python
} // namespace proxddp

