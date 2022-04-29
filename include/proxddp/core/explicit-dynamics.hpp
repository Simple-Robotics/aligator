#pragma once

#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include <proxnlp/manifold-base.hpp>


namespace proxddp
{

  // fwd declaration of ExplicitDynamicsDataTpl
  template<typename _Scalar>
  struct ExplicitDynamicsDataTpl;

  /** @brief Explicit forward dynamics model.
   * 
   * 
   *  @details    The forward dynamics are assumed to be of the format
   *              \f$ f(x, u) \ominus x' \f$.
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

    /// The constructor requires providing the next state's manifold.
    ExplicitDynamicsModelTpl(const int ndx1,
                             const int nu,
                             const Manifold& space2)
      : DynamicsModelTpl<Scalar>(ndx1, nu, space2.ndx())
      , space2_(space2) {}

    ExplicitDynamicsModelTpl(const Manifold& space,
                             const int nu)
      : ExplicitDynamicsModelTpl(space.ndx(), nu, space) {}

    /// Evaluate the forward dynamics.
    void virtual forward(const ConstVectorRef& x,
                         const ConstVectorRef& u,
                         VectorRef out) const = 0;

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
      space2_.difference(y, d.xout_, d.value_);
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
      space2_.Jdifference(y, d.xout_, Jtemp, 0);
      d.Jx_ = Jtemp * d.Jx_;
      space2_.Jdifference(y, d.xout_, Jtemp, 1);
      d.Ju_ = Jtemp * d.Ju_;
      d.Jy_ = -MatrixXs::Identity(this->ndx2);
    }


    std::shared_ptr<Data> createData() const
    {
      return std::make_shared<SpecificData>(this->ndx1, this->nu, this->ndx2);
    }

  protected:
    const Manifold& space2_;
  };

  /// @brief    Specific data struct for explicit dynamics.
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
      this->Jy_ = -MatrixXs::Identity(ndx2);
    }
    
  };
  
} // namespace proxddp

#include "proxddp/core/explicit-dynamics.hxx"
