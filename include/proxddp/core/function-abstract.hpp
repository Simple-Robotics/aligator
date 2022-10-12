/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
/// @file function-abstract.hpp
/// @brief  Base definitions for ternary functions.
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/clone.hpp"

#include <fmt/format.h>
#include <ostream>

namespace proxddp {

/// @brief    Class representing ternary functions \f$f(x,u,x')\f$.
template <typename _Scalar> struct StageFunctionTpl {
public:
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);
  using Data = FunctionDataTpl<Scalar>;

  /// @brief Current state dimension
  const int ndx1;
  /// @brief Control dimension
  const int nu;
  /// @brief Next state dimension
  const int ndx2;
  /// @brief Function codimension
  const int nr;

  StageFunctionTpl(const int ndx1, const int nu, const int ndx2, const int nr);

  StageFunctionTpl(const int ndx, const int nu, const int nr);

  /**
   * @brief       Evaluate the function.
   * @param x     Current state.
   * @param u     Controls.
   * @param y     Next state.
   * @param data  Data holding struct.
   */
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const = 0;

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
                                const ConstVectorRef &u,
                                const ConstVectorRef &y, Data &data) const = 0;

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
                                            const ConstVectorRef &y,
                                            const ConstVectorRef &lbda,
                                            Data &data) const;

  virtual ~StageFunctionTpl() = default;

  /// @brief Instantiate a Data object.
  virtual shared_ptr<Data> createData() const;
};

/// @brief  Struct to hold function data.
template <typename _Scalar>
struct FunctionDataTpl : Cloneable<FunctionDataTpl<_Scalar>> {
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const int ndx1;
  const int nu;
  const int ndx2;
  const int nr;
  /// @brief Total number of variables.
  const int nvar = ndx1 + nu + ndx2;

  /// Function value.
  VectorXs value_;
  /// Full Jacobian.
  MatrixXs jac_buffer_;
  /// Vector-Hessian product buffer.
  MatrixXs vhp_buffer_;
  /// Jacobian with respect to \f$x\f$.
  MatrixRef Jx_;
  /// Jacobian with respect to \f$u\f$.
  MatrixRef Ju_;
  /// Jacobian with respect to \f$y\f$.
  MatrixRef Jy_;

  /* Vector-Hessian product buffers */

  MatrixRef Hxx_;
  MatrixRef Hxu_;
  MatrixRef Hxy_;
  MatrixRef Huu_;
  MatrixRef Huy_;
  MatrixRef Hyy_;

  /// @brief Default constructor.
  FunctionDataTpl(const int ndx1, const int nu, const int ndx2, const int nr);
  virtual ~FunctionDataTpl() = default;

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const FunctionDataTpl<T> &self);
};

} // namespace proxddp

#include "proxddp/core/function-abstract.hxx"
