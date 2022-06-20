#pragma once

#include "proxddp/core/function.hpp"


namespace proxddp
{
  /** @brief Linear function \f$f(x,u,y) = Ax + Bu + Cy + d\f$.
   * 
   */
  template<typename Scalar>
  struct LinearFunction : StageFunctionTpl<Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Data = FunctionDataTpl<Scalar>;

    MatrixXs A_;
    MatrixXs B_;
    MatrixXs C_;
    VectorXs d_;

    LinearFunction(const int ndx, const int nu, const int ndx2, const int nr)
      : StageFunctionTpl<Scalar>(ndx, nu, ndx2, nr)
      , A_(nr, ndx), B_(nr, nu), C_(nr, ndx2), d_(nr)
    {
      A_.setZero();
      B_.setZero();
      C_.setZero();
      d_.setZero();
    }

    LinearFunction(const ConstMatrixRef A, const ConstMatrixRef B, const ConstMatrixRef C, const ConstVectorRef d)
      : StageFunctionTpl<Scalar>(A.cols(), B.cols(), C.cols(), d.rows())
      , A_(A), B_(B), C_(C), d_(d)
    {
      assert((A_.rows() == d_.cols()) &&
             (A_.rows() == B_.rows()) &&
             (A_.rows() == C_.rows()) &&
             "Number of rows not consistent.");
    }

    /// @brief Constructor where \f$C = 0\f$ is assumed.
    LinearFunction(const ConstMatrixRef A, const ConstMatrixRef B, const ConstVectorRef d)
      : LinearFunction(A, B, MatrixXs::Zero(A.rows(), A.cols()), d)
      {}

    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  Data& data) const
    {
      data.value_ = A_ * x + B_ * u + C_ * y + d_;
    }

    void computeJacobians(const ConstVectorRef&,
                          const ConstVectorRef&,
                          const ConstVectorRef&,
                          Data& data) const
    {
      data.Jx_ = A_;
      data.Ju_ = B_;
      data.Jy_ = C_;
    }
  };
  
} // namespace proxddp

