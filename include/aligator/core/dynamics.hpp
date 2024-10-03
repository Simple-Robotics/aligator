/// @file dynamics.hpp
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"
#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>

namespace aligator {
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
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = DynamicsDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using PolyManifold = xyz::polymorphic<Manifold>;

  /// State space for the input.
  PolyManifold space_;
  /// State space for the output of this dynamics model.
  PolyManifold space_next_;

  /// @copybrief space_
  const Manifold &space() const { return *space_; }
  /// @copybrief space_next_
  const Manifold &space_next() const { return *space_next_; }
  /// @brief Check if this dynamics model is implicit or explicit.
  virtual bool isExplicit() const { return false; }

  inline int nx1() const { return space_->nx(); }
  inline int nx2() const { return space_next_->nx(); }

  /**
   * @brief  Constructor for dynamics.
   *
   * @param   space State space.
   * @param   nu    Control dimension
   * @param   ndx2  Next state space dimension.
   */
  DynamicsModelTpl(PolyManifold space, const int nu);

  /**
   * @copybrief DynamicsModelTpl This constructor assumes same dimension for the
   * current and next state.
   *
   * @param   space State space for the current, and next node.
   * @param   nu    Control dimension
   */
  DynamicsModelTpl(PolyManifold space, const int nu, PolyManifold space2);

  virtual void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, Data &) const;

  virtual void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                                const ConstVectorRef &, Data &) const;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/dynamics.txx"
#endif
