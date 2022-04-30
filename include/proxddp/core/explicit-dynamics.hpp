#pragma once

#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include <proxnlp/manifold-base.hpp>


namespace proxddp
{

  // fwd declaration of ExplicitDynamicsDataTpl
  template<typename _Scalar>
  struct ExplicitDynamicsDataTpl;

  /** @brief Explicit forward dynamics model \f$ x_{k+1} = f(x_k, u_k) \f$.
   * 
   * 
   *  @details    Forward dynamics \f$ x_{k+1} = f(x_k, u_k) \f$.
   *              The corresponding residuals for multiple-shooting formulations are
   *              \f$ f(x_k, u_k) \ominus x_{k+1} \f$.
   */
  template<typename _Scalar>
  struct ExplicitDynamicsModelTpl : DynamicsModelTpl<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)
    using Data = FunctionDataTpl<Scalar>;
    /// Specific data struct for this type
    using SpecificData = ExplicitDynamicsDataTpl<Scalar>;
    using Manifold = ManifoldAbstractTpl<Scalar>;

    const Manifold& out_space_;

    /// The constructor requires providing the next state's manifold.
    ExplicitDynamicsModelTpl(const int ndx1,
                             const int nu,
                             const Manifold& space2)
      : DynamicsModelTpl<Scalar>(ndx1, nu, space2.ndx())
      , out_space_(space2) {}

    ExplicitDynamicsModelTpl(const Manifold& space,
                             const int nu)
      : ExplicitDynamicsModelTpl(space.ndx(), nu, space) {}

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
                  Data& data) const
    {
      // Call the forward dynamics and set the function residual
      // value to the difference between y and the xout_.
      auto d = static_cast<SpecificData&>(data);
      this->forward(x, u, d.xout_);
      out_space_.difference(y, d.xout_, d.value_);
    }

    void computeJacobians(const ConstVectorRef& x,
                          const ConstVectorRef& u,
                          const ConstVectorRef& y,
                          Data& data) const
    {
      SpecificData& d = static_cast<SpecificData&>(data);
      this->dForward(x, u, d.Jx_, d.Ju_); // dxnext_(x,u)
      // compose by jacobians of log (xout - y)
      MatrixXs Jtemp(this->ndx2, this->ndx2);
      out_space_.Jdifference(y, d.xout_, Jtemp, 0);
      d.Jx_ = Jtemp * d.Jx_;
      out_space_.Jdifference(y, d.xout_, Jtemp, 1);
      d.Ju_ = Jtemp * d.Ju_;
      d.Jy_ = -MatrixXs::Identity(this->ndx2, this->ndx2);
    }


    std::shared_ptr<Data> createData() const
    {
      return std::make_shared<SpecificData>(this->ndx1, this->nu, this->ndx2);
    }

  };

  /// @brief    Specific data struct for explicit dynamics ExplicitDynamicsModelTpl.
  template<typename _Scalar>
  struct ExplicitDynamicsDataTpl : FunctionDataTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)
    VectorXs xout_;

    ExplicitDynamicsDataTpl(const int ndx1,
                            const int nu,
                            const int ndx2)
      : FunctionDataTpl<Scalar>(ndx1, nu, ndx2, ndx2)
      , xout_(ndx2)
    {
      xout_.setZero();
      this->Jy_ = -MatrixXs::Identity(ndx2, ndx2);
    }
    
    ExplicitDynamicsDataTpl(const int ndx, const int nu)
      : ExplicitDynamicsDataTpl(ndx, nu, ndx) {}

  };
  
} // namespace proxddp

#include "proxddp/core/explicit-dynamics.hxx"
