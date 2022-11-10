#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {

template <typename Scalar>
struct LinearFunctionCompositionTpl : StageFunctionTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = FunctionDataTpl<Scalar>;

  shared_ptr<Base> func;
  MatrixXs A;
  VectorXs b;

  struct OwnData : Data {
    shared_ptr<Data> sub_data;
    OwnData(const LinearFunctionCompositionTpl *ptr)
        : Data(ptr->ndx1, ptr->nu, ptr->ndx2, ptr->nr),
          sub_data(ptr->func->createData()) {}
  };

  LinearFunctionCompositionTpl(shared_ptr<Base> func, const ConstMatrixRef A,
                               const ConstVectorRef b);

  LinearFunctionCompositionTpl(shared_ptr<Base> func, const ConstMatrixRef A);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const;

  shared_ptr<Data> createData() const;
};

} // namespace proxddp

#include "proxddp/modelling/linear-function-composition.hxx"
