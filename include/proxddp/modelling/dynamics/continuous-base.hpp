#pragma once

#include "proxddp/modelling/dynamics/fwd.hpp"


namespace proxddp
{
namespace dynamics
{

  /** @brief Continuous dynamics described by differential-algebraic equations (DAEs).
   * 
   * @details Continuous dynamics described as \f$ f(x, u, \dot{x}) = 0 \f$.
   *          The codimension of this function is the same as that of \f$x\f$.
   */
  template<typename _Scalar>
  struct ContinuousDynamicsTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
    using Manifold = ManifoldAbstractTpl<Scalar>;
    /// State space.
    const Manifold& space_;
    /// State space dimension.
    const int ndx_;
    /// Control space dimension.
    const int nu_;

    inline int ndx() const { return ndx_; }
    inline int nu()  const { return nu_; }

    ContinuousDynamicsTpl(const Manifold& manifold, const int nu)
      : space_(manifold)
      , ndx_(manifold.ndx())
      , nu_(nu)
      {}

    virtual ~ContinuousDynamicsTpl() = default;

    /// @brief   Evaluate the vector field at a point \f$(x, u)\f$.
    /// @param   x The input state variable.
    /// @param   u The input control variable.
    /// @param[out] out The output vector.
    virtual void forward(const ConstVectorRef& x, const ConstVectorRef& u, const ConstVectorRef& xdot, VectorRef out) const = 0;

    /// @brief  Differentiate the vector field.
    virtual void dForward(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& xdot,
                          MatrixRef Jx,
                          MatrixRef Ju) const = 0;

    using Data = ContinuousDynamicsDataTpl<Scalar>;
    /// @brief  Create a data holder instance.
    shared_ptr<Data> createData() const
    {
      return std::make_shared<Data>(ndx_, nu_);
    }

  };

  /// @brief  Data struct for ContinuousDynamicsTpl.
  template<typename _Scalar>
  struct ContinuousDynamicsDataTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

    VectorXs value_;
    MatrixXs Jx_;
    MatrixXs Ju_;
    MatrixXs Jxdot_;

    ContinuousDynamicsDataTpl(const int ndx, const int nu)
      : value_(ndx)
      , Jx_(ndx, ndx)
      , Ju_(ndx, nu)
      , Jxdot_(ndx, ndx)
    {
      value_.setZero();
      Jx_.setZero();
      Ju_.setZero();
      Jxdot_.setZero();
    }

  };

} // namespace dynamics  
} // namespace proxddp

