#pragma once

#include "aligator/core/function-abstract.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace aligator {

template <typename _Scalar>
struct GravityCompensationResidualTpl : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = StageFunctionDataTpl<Scalar>;
  using Model = pinocchio::ModelTpl<Scalar>;

  Model pin_model_;
  MatrixXs actuation_matrix_;
  bool use_actuation_matrix;

  /// Constructor with an actuation matrix
  GravityCompensationResidualTpl(int ndx, const MatrixXs &actuation_matrix,
                                 const Model &model);
  /// Full actuation constructor
  GravityCompensationResidualTpl(int ndx, const Model &model);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;
  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const;

  struct Data : BaseData {
    pinocchio::DataTpl<Scalar> pin_data_;
    VectorXs tmp_torque_;
    MatrixXs gravity_partial_dq_;
    Data(const GravityCompensationResidualTpl &resdl);
  };

  shared_ptr<BaseData> createData() const;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct GravityCompensationResidualTpl<context::Scalar>;
#endif

} // namespace aligator
