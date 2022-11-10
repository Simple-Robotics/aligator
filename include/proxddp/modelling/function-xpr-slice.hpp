/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"

namespace proxddp {

/// @brief Slicing and indexing of a function's output.
template <typename Scalar>
struct FunctionSliceXprTpl : StageFunctionTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using BaseData = FunctionDataTpl<Scalar>;

  struct OwnData : BaseData {
    /// @brief Base residual's data object.
    shared_ptr<BaseData> sub_data;
    VectorXs lbda_sub;

    OwnData(FunctionSliceXprTpl<Scalar> const *obj)
        : BaseData(obj->ndx1, obj->nu, obj->ndx2, obj->nr),
          sub_data(obj->func->createData()), lbda_sub(obj->nr) {}
  };

  using Base = StageFunctionTpl<Scalar>;

  shared_ptr<Base> func;
  /// @brief
  std::vector<int> indices;

  FunctionSliceXprTpl(shared_ptr<Base> func, std::vector<int> indices)
      : Base(func->ndx1, func->nu, func->ndx2, (int)indices.size()), func(func),
        indices(indices) {}

  FunctionSliceXprTpl(shared_ptr<Base> func, int idx)
      : FunctionSliceXprTpl(func, std::vector<int>({idx})) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const {
    OwnData &d = static_cast<OwnData &>(data);
    assert(d.sub_data != 0);
    BaseData &sub_data = *d.sub_data;
    // evaluate base
    func->evaluate(x, u, y, sub_data);

    for (long j = 0; j < (long)indices.size(); j++) {
      int i = indices[j];
      d.value_(j) = sub_data.value_(i);
    }
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const {
    OwnData &d = static_cast<OwnData &>(data);
    assert(d.sub_data != 0);
    BaseData &sub_data = *d.sub_data;
    // evaluate base
    func->computeJacobians(x, u, y, sub_data);

    for (long j = 0; j < (long)indices.size(); j++) {
      int i = indices[j];
      d.jac_buffer_.row(j) = sub_data.jac_buffer_.row(i);
    }
  }

  void computeVectorHessianProducts(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
                                    const ConstVectorRef &lbda,
                                    BaseData &data) const {
    OwnData &d = static_cast<OwnData &>(data);
    assert(d.sub_data != 0);
    BaseData &sub_data = *d.sub_data;

    d.lbda_sub = lbda;
    for (long j = 0; j < (long)indices.size(); j++) {
      d.lbda_sub(indices[j]) = 0.;
    }

    func->computeVectorHessianProducts(x, u, y, d.lbda_sub, sub_data);
  }

  shared_ptr<BaseData> createData() const {
    return std::static_pointer_cast<BaseData>(std::make_shared<OwnData>(this));
  }
};

} // namespace proxddp
