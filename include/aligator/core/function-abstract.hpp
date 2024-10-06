/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
/// @file function-abstract.hpp
/// @brief  Base definitions for ternary functions.
#pragma once

#include "aligator/fwd.hpp"
#include <ostream>

namespace aligator {

/// @brief    Class representing ternary functions \f$f(x,u,x')\f$.
template <typename _Scalar> struct StageFunctionTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = StageFunctionDataTpl<Scalar>;

  /// @brief Current state dimension
  const int ndx1;
  /// @brief Control dimension
  const int nu;
  /// @brief Function codimension
  const int nr;

  StageFunctionTpl(const int ndx, const int nu, const int nr);

  /**
   * @brief       Evaluate the function.
   * @param x     Current state.
   * @param u     Controls.
   * @param data  Data holding struct.
   */
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        Data &data) const = 0;

  /** @brief    Compute Jacobians of this function.
   *
   * @details   This computes the Jacobians
   * \f$
   *   (\frac{\partial f}{\partial x},
   *   \frac{\partial f}{\partial u},
   *   \frac{\partial f}{\partial x'})
   * \f$
   *
   * @param x     Current state.
   * @param u     Controls.
   * @param y     Next state.
   * @param data  Data holding struct.
   */
  virtual void computeJacobians(const ConstVectorRef &x,
                                const ConstVectorRef &u, Data &data) const = 0;

  /** @brief    Compute the vector-hessian products of this function.
   *
   *  @param x     Current state.
   *  @param u     Controls.
   *  @param y     Next state.
   *  @param lbda Multiplier estimate.
   *  @param data  Data holding struct.
   */
  virtual void computeVectorHessianProducts(const ConstVectorRef &x,
                                            const ConstVectorRef &u,
                                            const ConstVectorRef &lbda,
                                            Data &data) const;

  virtual ~StageFunctionTpl() = default;

  /// @brief Instantiate a Data object.
  virtual shared_ptr<Data> createData() const;
};

/// @brief  Base struct for function data.
template <typename _Scalar> struct StageFunctionDataTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  const int ndx1;
  const int nu;
  const int nr;
  /// @brief Total number of variables.
  const int nvar = ndx1 + nu;

  /// Function value.
  VectorXs value_;
  VectorRef valref_;
  /// Full Jacobian.
  MatrixXs jac_buffer_;
  /// Vector-Hessian product buffer.
  MatrixXs vhp_buffer_;
  /// Jacobian with respect to \f$x\f$.
  MatrixRef Jx_;
  /// Jacobian with respect to \f$u\f$.
  MatrixRef Ju_;

  /* Vector-Hessian product buffers */

  MatrixRef Hxx_;
  MatrixRef Hxu_;
  MatrixRef Huu_;

  /// @brief Default constructor.
  StageFunctionDataTpl(const int ndx, const int nu, const int nr);
  StageFunctionDataTpl(const StageFunctionTpl<Scalar> &model)
      : StageFunctionDataTpl(model.ndx1, model.nu, model.nr) {}
  virtual ~StageFunctionDataTpl() = default;
};

template <typename T>
std::ostream &operator<<(std::ostream &oss,
                         const StageFunctionDataTpl<T> &self);

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/function-abstract.txx"
#endif
