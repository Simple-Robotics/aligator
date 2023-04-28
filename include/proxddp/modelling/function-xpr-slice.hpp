/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {

/// @brief Slicing and indexing of a function's output.
template <typename Scalar>
struct FunctionSliceXprTpl : StageFunctionTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = FunctionDataTpl<Scalar>;
  using Base = StageFunctionTpl<Scalar>;

  struct Data;

  shared_ptr<Base> func;
  /// @brief
  std::vector<int> indices;

  FunctionSliceXprTpl(shared_ptr<Base> func, std::vector<int> const &indices);

  FunctionSliceXprTpl(shared_ptr<Base> func, int idx);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const;

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
                                    const ConstVectorRef &lbda,
                                    BaseData &data) const;

  shared_ptr<BaseData> createData() const;
};

template <typename Scalar> struct FunctionSliceXprTpl<Scalar>::Data : BaseData {
  /// @brief Base residual's data object.
  shared_ptr<BaseData> sub_data;
  VectorXs lbda_sub;

  Data(FunctionSliceXprTpl<Scalar> const &obj)
      : BaseData(obj.ndx1, obj.nu, obj.ndx2, obj.nr),
        sub_data(obj.func->createData()), lbda_sub(obj.nr) {}
};

} // namespace proxddp

#include "proxddp/modelling/function-xpr-slice.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/modelling/function-xpr-slice.txx"
#endif
