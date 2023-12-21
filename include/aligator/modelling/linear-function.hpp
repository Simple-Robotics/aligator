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
  MatrixXs C_;
  VectorXs d_;

  LinearFunctionTpl(const int ndx, const int nu, const int ndx2, const int nr)
      : Base(ndx, nu, ndx2, nr), A_(nr, ndx), B_(nr, nu), C_(nr, ndx2), d_(nr) {
    A_.setZero();
    B_.setZero();
    C_.setZero();
    d_.setZero();
  }

  LinearFunctionTpl(const ConstMatrixRef A, const ConstMatrixRef B,
                    const ConstMatrixRef C, const ConstVectorRef d)
      : Base((int)A.cols(), (int)B.cols(), (int)C.cols(), (int)d.rows()), A_(A),
        B_(B), C_(C), d_(d) {
    assert((A_.rows() == d_.rows()) && (B_.rows() == d_.rows()) &&
           (C_.rows() == d_.rows()) && "Number of rows not consistent.");
  }

  /// @brief Constructor where \f$C = 0\f$ is assumed.
  LinearFunctionTpl(const ConstMatrixRef A, const ConstMatrixRef B,
                    const ConstVectorRef d)
      : LinearFunctionTpl(A, B, MatrixXs::Zero(A.rows(), A.cols()), d) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const {
    data.value_ = A_ * x + B_ * u + C_ * y + d_;
  }

  /**
   * @copybrief Base::computeJacobians()
   * @details   This implementation does nothing: the values of the Jacobians
   * are already set in createData().
   */
  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, Data &data) const {
    data.Jx_ = A_;
    data.Ju_ = B_;
    data.Jy_ = C_;
  }

  /// @copybrief Base::createData()
  /// @details   This override sets the appropriate values of the Jacobians.
  virtual shared_ptr<Data> createData() const {
    auto data =
        std::make_shared<Data>(this->ndx1, this->nu, this->ndx2, this->nr);
    data->Jx_ = A_;
    data->Ju_ = B_;
    data->Jy_ = C_;
    return data;
  }
};

} // namespace aligator
