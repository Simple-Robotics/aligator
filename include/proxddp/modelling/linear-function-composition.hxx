#pragma once

namespace proxddp {

template <typename Scalar>
LinearFunctionCompositionTpl<Scalar>::LinearFunctionCompositionTpl(
    shared_ptr<Base> func, const ConstMatrixRef A, const ConstVectorRef b)
    : Base(func->ndx1, func->nu, func->ndx2, (int)A.rows()), func(func), A(A),
      b(b) {
  if (func == 0) {
    PROXDDP_RUNTIME_ERROR("Underlying function cannot be nullptr.");
  }
  if (A.rows() != b.rows()) {
    PROXDDP_RUNTIME_ERROR("Incompatible dimensions: A.rows() != b.rows()");
  }
  if (A.cols() != func->nr) {
    PROXDDP_RUNTIME_ERROR("Incompatible dimensions: A.cols() != func.nr");
  }
}

template <typename Scalar>
LinearFunctionCompositionTpl<Scalar>::LinearFunctionCompositionTpl(
    shared_ptr<Base> func, const ConstMatrixRef A)
    : LinearFunctionCompositionTpl(func, A, VectorXs::Zero(A.rows())) {}

template <typename Scalar>
void LinearFunctionCompositionTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                    const ConstVectorRef &u,
                                                    const ConstVectorRef &y,
                                                    Data &data) const {
  OwnData &d = static_cast<OwnData &>(data);

  func->evaluate(x, u, y, *d.sub_data);
  data.value_ = A * d.sub_data->value_ + b;
}

template <typename Scalar>
void LinearFunctionCompositionTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    Data &data) const {
  OwnData &d = static_cast<OwnData &>(data);

  func->computeJacobians(x, u, y, *d.sub_data);
  data.jac_buffer_ = A * d.sub_data->jac_buffer_;
}

template <typename Scalar>
shared_ptr<FunctionDataTpl<Scalar>>
LinearFunctionCompositionTpl<Scalar>::createData() const {
  return shared_ptr<Data>(new OwnData(this));
}

} // namespace proxddp
