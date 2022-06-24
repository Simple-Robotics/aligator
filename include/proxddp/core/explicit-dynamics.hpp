#pragma once

#include "proxddp/core/dynamics.hpp"

#include <proxnlp/manifold-base.hpp>


namespace proxddp
{

  /** @brief Explicit forward dynamics model \f$ x_{k+1} = f(x_k, u_k) \f$.
   * 
   * 
   *  @details    Forward dynamics \f$ x_{k+1} = f(x_k, u_k) \f$.
   *              The corresponding residuals for multiple-shooting formulations are
   *  \f[
   *    \bar{f}(x_k, u_k, x_{k+1}) = f(x_k, u_k) \ominus x_{k+1}.
   *  \f]
   */
  template<typename _Scalar>
  struct ExplicitDynamicsModelTpl : DynamicsModelTpl<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar);
    using Base = DynamicsModelTpl<Scalar>;
    using BaseData = DynamicsDataTpl<Scalar>;
    using ExplicitData = ExplicitDynamicsDataTpl<Scalar>;
    using Manifold = ManifoldAbstractTpl<Scalar>;

    shared_ptr<Manifold> next_state_;

    /// @return Reference to output state space.
    const Manifold& out_space() const
    {
      return *next_state_;
    }

    /// The constructor requires providing the next state's manifold.
    ExplicitDynamicsModelTpl(const int ndx1,
                             const int nu,
                             const shared_ptr<Manifold>& next_state)
      : Base(ndx1, nu, next_state->ndx())
      , next_state_(next_state)
      {}

    ExplicitDynamicsModelTpl(const shared_ptr<Manifold>& next_state,
                             const int nu)
      : ExplicitDynamicsModelTpl(next_state->ndx(), nu, next_state)
      {}

    /// @brief Evaluate the forward discrete dynamics.
    void virtual forward(const ConstVectorRef& x,
                         const ConstVectorRef& u,
                         ExplicitData& data) const = 0;

    /// @brief Compute the Jacobians of the forward dynamics.
    void virtual dForward(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          ExplicitData& data) const = 0;

    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  BaseData& data) const;

    void computeJacobians(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& y,
                          BaseData& data) const;

    shared_ptr<BaseData> createData() const;

  };

  /// @brief    Specific data struct for explicit dynamics ExplicitDynamicsModelTpl.
  template<typename _Scalar>
  struct ExplicitDynamicsDataTpl : FunctionDataTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar);
    VectorXs xout_;
    VectorXs dx_;
    MatrixXs Jtemp_;

    VectorRef xoutref_;
    VectorRef dxref_;

    ExplicitDynamicsDataTpl(const int ndx1,
                            const int nu,
                            const ManifoldAbstractTpl<Scalar>& output_space)
      : FunctionDataTpl<Scalar>(ndx1, nu, output_space.ndx(), output_space.ndx())
      , xout_(output_space.neutral())
      , dx_(output_space.ndx())
      , Jtemp_(output_space.ndx(), output_space.ndx())
      , xoutref_(xout_)
      , dxref_(dx_)
    {
      xout_.setZero();
      dx_.setZero();
      Jtemp_.setZero();
    }

  };
  
} // namespace proxddp

#include "proxddp/core/explicit-dynamics.hxx"
