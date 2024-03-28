/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
/// @file function-abstract.hpp
/// @brief  Base definitions for ternary functions.
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/clone.hpp"
#include "aligator/core/common-model-builder-container.hpp"
#include "aligator/core/common-model-data-container.hpp"

#include <fmt/format.h>
#include <ostream>

namespace aligator {

/// @brief    Class representing ternary functions \f$f(x,u,x')\f$.
template <typename _Scalar>
struct StageFunctionTpl
    : std::enable_shared_from_this<StageFunctionTpl<_Scalar>> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Data = StageFunctionDataTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  /// @brief Current state dimension
  const int ndx1;
  /// @brief Control dimension
  const int nu;
  /// @brief Next state dimension
  const int ndx2;
  /// @brief Function codimension
  const int nr;

  StageFunctionTpl(const int ndx1, const int nu, const int ndx2, const int nr);

  /// Constructor where ndx2 = ndx1.
  StageFunctionTpl(const int ndx, const int nu, const int nr);

  /// @brief Create and configure CommonModelTpl
  virtual void configure(ALIGATOR_MAYBE_UNUSED CommonModelBuilderContainer
                             &common_buider_container) const {}

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

  /// @brief Instantiate a Data object with CommonModelData.
  /// By default, call createData()
  virtual shared_ptr<Data>
  createData(const CommonModelDataContainer &container) const;

  /// @brief Instantiate a Data object.
  virtual shared_ptr<Data> createData() const;
};

/// @brief  Base struct for function data.
template <typename _Scalar>
struct StageFunctionDataTpl : Cloneable<StageFunctionDataTpl<_Scalar>> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const int ndx1;
  const int nu;
  const int ndx2;
  const int nr;
  /// @brief Total number of variables.
  const int nvar = ndx1 + nu + ndx2;

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
  StageFunctionDataTpl(const int ndx1, const int nu, const int ndx2,
                       const int nr);
  virtual ~StageFunctionDataTpl() = default;

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const StageFunctionDataTpl<T> &self);

protected:
  virtual StageFunctionDataTpl *clone_impl() const {
    return new StageFunctionDataTpl(*this);
  }
};

#define ALIGATOR_STAGE_FUNCTION_TYPEDEFS(Scalar, _Data)                        \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using Base = StageFunctionTpl<Scalar>;                                       \
  using BaseData = StageFunctionDataTpl<Scalar>;                               \
  using Data = _Data<Scalar>;                                                  \
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;  \
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>

#define ALIGATOR_STAGE_FUNCTION_DATA_TYPEDEFS(Scalar, _Model)                  \
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);                                           \
  using Model = _Model<Scalar>;                                                \
  using Base = StageFunctionDataTpl<Scalar>;                                   \
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>

} // namespace aligator

#include "aligator/core/function-abstract.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/function-abstract.txx"
#endif
