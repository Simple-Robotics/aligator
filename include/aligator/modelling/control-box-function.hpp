#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {
///
/// @brief   A simple function \f$f(u) = [u_{\min} - u; u - u_{\max}]\f$.
///
/// @note This function was initially meant to be used along with
/// NegativeOrthantTpl to create control bound constraints.
/// For this purpose, please use BoxConstraintTpl instead.
/// @deprecated This class is not meant to be used anymore, and has been
/// deprecated. It might be removed in the future. Use BoxContraintTpl to
/// implement control bounds. See related note.
template <typename _Scalar>
struct [[deprecated("ControlBoxFunction should not be used. Instead, just use "
                    "the identity function and a BoxConstraintTpl.")]]
ControlBoxFunctionTpl : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = StageFunctionDataTpl<Scalar>;

  VectorXs umin_, umax_;

  /// @brief Standard constructor: takes state space dimension and desired
  /// control bounds as vectors.
  ControlBoxFunctionTpl(const int ndx, const VectorXs &umin,
                        const VectorXs &umax);

  /// @brief Constructor which takes control bounds as scalars cast into vectors
  /// of appropriate dimension.
  ControlBoxFunctionTpl(const int ndx, const int nu, const Scalar umin,
                        const Scalar umax);

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                Data &data) const override;

  /**
   * @copybrief Base::computeJacobians()
   * @details   This implementation does nothing: the values of the Jacobians
   * are already set in createData().
   */
  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        Data &data) const override;

  /// @copybrief Base::createData()
  /// @details   This override sets the appropriate values of the Jacobians.
  virtual shared_ptr<Data> createData() const override;
};

} // namespace aligator

#include "aligator/modelling/control-box-function.hxx"
