/// @file dynamics.hpp
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
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
template <typename _Scalar> struct DynamicsModelTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = DynamicsDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  /// State space for the input.
  xyz::polymorphic<Manifold> space_;
  /// Control dimension
  /// State space for the output of this dynamics model.
  xyz::polymorphic<Manifold> space_next_;
  const int ndx1;
  const int nu;
  const int ndx2;

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
  DynamicsModelTpl(xyz::polymorphic<Manifold> space, const int nu);

  /**
   * @copybrief DynamicsModelTpl This constructor assumes same dimension for the
   * current and next state.
   *
   * @param   space      State space for the current node.
   * @param   nu         Control dimension
   * @param   space_next State space for the next node.
   */
  DynamicsModelTpl(xyz::polymorphic<Manifold> space, const int nu,
                   xyz::polymorphic<Manifold> space_next);

  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xn, Data &) const = 0;

  virtual void computeJacobians(const ConstVectorRef &x,
                                const ConstVectorRef &u,
                                const ConstVectorRef &xn, Data &) const = 0;

  virtual void computeVectorHessianProducts(const ConstVectorRef &x,
                                            const ConstVectorRef &u,
                                            const ConstVectorRef &xn,
                                            const ConstVectorRef &lbda,
                                            Data &data) const;

  virtual shared_ptr<Data> createData() const;

  virtual ~DynamicsModelTpl() = default;
};

template <typename _Scalar> struct DynamicsDataTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  const int ndx1;
  const int nu;
  const int ndx2;
  /// @brief Total number of variables.
  const int nvar = ndx1 + nu + ndx2;

  /// Function value.
  VectorXs value_;
  VectorRef valref_;
  /// Full Jacobian.
  MatrixXs jac_buffer_;
  /// Jacobian with respect to \f$x\f$.
  MatrixRef Jx_;
  /// Jacobian with respect to \f$u\f$.
  MatrixRef Ju_;
  /// Jacobian with respect to \f$y\f$.
  MatrixRef Jy_;

  /* Vector-Hessian product buffers */

  MatrixXs Hxx_;
  MatrixXs Hxu_;
  MatrixXs Hxy_;
  MatrixXs Huu_;
  MatrixXs Huy_;
  MatrixXs Hyy_;

  DynamicsDataTpl(const DynamicsModelTpl<Scalar> &model);
  DynamicsDataTpl(const int ndx1, const int nu, const int ndx2);
  virtual ~DynamicsDataTpl() = default;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/dynamics.txx"
#endif
