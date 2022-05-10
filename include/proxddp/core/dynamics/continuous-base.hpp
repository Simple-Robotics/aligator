#pragma once
/// @file continuous-base.hpp
/// @brief Base definitions for continuous dynamics.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include "proxddp/core/dynamics/fwd.hpp"


namespace proxddp
{
namespace dynamics
{

  /** @brief Continuous dynamics described by differential-algebraic equations (DAEs)
   *         \f$F(\dot{x}, x, u) = 0\f$.
   * 
   * @details Continuous dynamics described as \f$ f(x, u, \dot{x}) = 0 \f$.
   *          The codimension of this function is the same as that of \f$x\f$.
   */
  template<typename _Scalar>
  struct ContinuousDynamicsTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Manifold = ManifoldAbstractTpl<Scalar>;
    using Data = ContinuousDynamicsDataTpl<Scalar>;

    /// State space.
    const Manifold& space_;
    /// Control space dimension.
    const int nu_;

    inline int ndx() const { return space_.ndx(); }
    inline int nu()  const { return nu_; }

    ContinuousDynamicsTpl(const Manifold& space, const int nu)
      : space_(space)
      , nu_(nu)
      {}

    virtual ~ContinuousDynamicsTpl() = default;

    /// @brief   Evaluate the vector field at a point \f$(x, u)\f$.
    /// @param   x The input state variable.
    /// @param   u The input control variable.
    /// @param   xdot Derivative of the state.
    /// @param[out] data The output data object.
    virtual void evaluate(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& xdot,
                          Data& data) const = 0;

    /// @brief  Differentiate the vector field.
    /// @param   x The input state variable.
    /// @param   u The input control variable.
    /// @param   xdot Derivative of the state.
    /// @param[out] data The output data object.
    virtual void computeJacobians(const ConstVectorRef& x,
                                  const ConstVectorRef& u,
                                  const ConstVectorRef& xdot,
                                  Data& data) const = 0;

    /// @brief  Create a data holder instance.
    virtual shared_ptr<Data> createData() const
    {
      return std::make_shared<Data>(ndx(), nu());
    }

  };

  /// @brief  Data struct for ContinuousDynamicsTpl.
  template<typename _Scalar>
  struct ContinuousDynamicsDataTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    /// Residual value \f$e = f(x,u,\dot{x})\f$
    VectorXs error_;
    MatrixXs Jx_;
    MatrixXs Ju_;
    MatrixXs Jxdot_;

    ContinuousDynamicsDataTpl(const int ndx, const int nu)
      : error_(ndx)
      , Jx_(ndx, ndx)
      , Ju_(ndx, nu)
      , Jxdot_(ndx, ndx)
    {
      error_.setZero();
      Jx_.setZero();
      Ju_.setZero();
      Jxdot_.setZero();
    }

  };

} // namespace dynamics  
} // namespace proxddp

