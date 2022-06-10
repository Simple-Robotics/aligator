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
    using Data = FunctionDataTpl<Scalar>;
    /// Specific data struct for this type
    using SpecificData = ExplicitDynamicsDataTpl<Scalar>;
    using Manifold = ManifoldAbstractTpl<Scalar>;

    shared_ptr<const Manifold> out_space_;

    /// @return Reference to output state space.
    const Manifold& out_space() const
    {
      return *out_space_;
    }

    /// The constructor requires providing the next state's manifold.
    ExplicitDynamicsModelTpl(const int ndx1,
                             const int nu,
                             const shared_ptr<const Manifold>& space2)
      : DynamicsModelTpl<Scalar>(ndx1, nu, space2->ndx())
      , out_space_(space2) {}

    ExplicitDynamicsModelTpl(const shared_ptr<const Manifold>& space,
                             const int nu)
      : ExplicitDynamicsModelTpl(space->ndx(), nu, space) {}

    /// @brief Evaluate the forward discrete dynamics.
    void virtual forward(const ConstVectorRef& x,
                         const ConstVectorRef& u,
                         VectorRef out) const = 0;

    /// @brief Compute the Jacobians of the forward dynamics.
    void virtual dForward(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          MatrixRef Jx,
                          MatrixRef Ju) const = 0;

    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  Data& data) const override;

    void computeJacobians(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& y,
                          Data& data) const override;

    std::shared_ptr<Data> createData() const override
    {
      return std::make_shared<SpecificData>(this->ndx1, this->nu, this->out_space());
    }

  };

  /// @brief    Specific data struct for explicit dynamics ExplicitDynamicsModelTpl.
  template<typename _Scalar>
  struct ExplicitDynamicsDataTpl : FunctionDataTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar);
    VectorXs xout_;
    MatrixXs Jtemp_;

    ExplicitDynamicsDataTpl(const int ndx1,
                            const int nu,
                            const ManifoldAbstractTpl<Scalar>& output_space)
      : FunctionDataTpl<Scalar>(ndx1, nu, output_space.ndx(), output_space.ndx())
      , xout_(output_space.nx())
      , Jtemp_(output_space.ndx(), output_space.ndx())
    {
      xout_.setZero();
      Jtemp_.setZero();
    }

  };
  
} // namespace proxddp

#include "proxddp/core/explicit-dynamics.hxx"
