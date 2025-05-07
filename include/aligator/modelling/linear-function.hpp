#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {
/** @brief Linear function \f$f(x,u,y) = Ax + Bu + Cy + d\f$.
 *
 */
template <typename Scalar> struct LinearFunctionTpl : StageFunctionTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = StageFunctionDataTpl<Scalar>;

  MatrixXs A_;
  MatrixXs B_;
  VectorXs d_;

  LinearFunctionTpl(const int ndx, const int nu, const int nr);

  LinearFunctionTpl(const ConstMatrixRef A, const ConstMatrixRef B,
                    const ConstVectorRef d);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                Data &data) const override {
    data.value_ = d_;
    data.value_.noalias() += A_ * x;
    data.value_.noalias() += B_ * u;
  }

  /**
   * @copybrief Base::computeJacobians()
   * @details   This implementation does nothing: the values of the Jacobians
   * are already set in createData().
   */
  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        Data &data) const override {
    data.Jx_ = A_;
    data.Ju_ = B_;
  }

  /// @copybrief Base::createData()
  /// @details   This override sets the appropriate values of the Jacobians.
  virtual shared_ptr<Data> createData() const override {
    auto data = std::make_shared<Data>(this->ndx1, this->nu, this->nr);
    data->Jx_ = A_;
    data->Ju_ = B_;
    return data;
  }
};

template <typename Scalar>
LinearFunctionTpl<Scalar>::LinearFunctionTpl(const int ndx, const int nu,
                                             const int nr)
    : Base(ndx, nu, nr)
    , A_(nr, ndx)
    , B_(nr, nu)
    , d_(nr) {
  A_.setZero();
  B_.setZero();
  d_.setZero();
}

template <typename Scalar>
LinearFunctionTpl<Scalar>::LinearFunctionTpl(const ConstMatrixRef A,
                                             const ConstMatrixRef B,
                                             const ConstVectorRef d)
    : Base((int)A.cols(), (int)B.cols(), (int)d.rows())
    , A_(A)
    , B_(B)
    , d_(d) {
  assert((A_.rows() == d_.rows()) && (B_.rows() == d_.rows()) &&
         "Number of rows not consistent.");
}

} // namespace aligator
