/// @file dynamics.hpp
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {
/**
 * @brief   Dynamics model: describes system dynamics through an implicit
 * relation \f$f(x,u,x') = 0\f$.
 *
 * @details A dynamics model is a function  \f$f(x,u,x')\f$ that must be set to
 * zero, describing the dynamics mapping \f$(x, u) \mapsto x'\f$.
 *          DynamicsModelTpl::nr is assumed to be equal to
 * DynamicsModelTpl::ndx2.
 */
template <typename _Scalar>
struct DynamicsModelTpl : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Base::ndx1;
  using Base::ndx2;
  using Base::nu;

  using Manifold = ManifoldAbstractTpl<Scalar>;
  /// State space for the input.
  shared_ptr<Manifold> space_;
  /// State space for the output of this dynamics model; by default, the same
  /// space as the input.
  shared_ptr<Manifold> space_next_ = space_;

  /// @copybrief space_
  const Manifold &space() const { return *space_; }
  /// @copybrief space_next_
  const Manifold &space_next() const { return *space_next_; }

  /**
   * @brief  Constructor for dynamics.
   *
   * @param   space State space.
   * @param   nu    Control dimension
   * @param   ndx2  Next state space dimension.
   */
  DynamicsModelTpl(const shared_ptr<Manifold> &space, const int nu,
                   const int ndx2)
      : Base(space->ndx(), nu, ndx2, ndx2), space_(space) {}

  /**
   * @copybrief DynamicsModelTpl This constructor assumes same dimension for the
   * current and next state.
   *
   * @param   space State space for the current, and next node.
   * @param   nu    Control dimension
   */
  DynamicsModelTpl(const shared_ptr<Manifold> &space, const int nu)
      : DynamicsModelTpl<Scalar>(space, nu, space->ndx()) {}
};

} // namespace proxddp
