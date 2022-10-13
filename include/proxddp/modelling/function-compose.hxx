#pragma once

namespace proxddp {

template <typename Scalar>
LinearFunctionCompositionTpl<Scalar>::LinearFunctionCompositionTpl(
    shared_ptr<Base> func, const ConstMatrixRef A, const ConstVectorRef b)
    : Base(func->ndx1, func->nu, func->ndx2, (int)A.rows()), func(func), A(A),
      b(b) {
  if (func == 0) {
    proxddp_runtime_error("Underlying function cannot be nullptr.");
  }
  if (A.rows() != b.rows()) {
    proxddp_runtime_error("Incompatible dimensions: A.rows() != b.rows()");
  }
  if (A.cols() != func->nr) {
    proxddp_runtime_error("Incompatible dimensions: A.cols() != func.nr");
  }
}

template <typename Scalar>
LinearFunctionCompositionTpl<Scalar>::LinearFunctionCompositionTpl(
    shared_ptr<Base> func, const ConstMatrixRef A)
    : LinearFunctionCompositionTpl(func, A, VectorXs::Zero(A.rows())) {}

} // namespace proxddp
