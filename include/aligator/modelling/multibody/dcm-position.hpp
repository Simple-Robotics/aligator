#pragma once

#include "aligator/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>

namespace aligator {

template <typename Scalar> struct DCMPositionDataTpl;

template <typename _Scalar>
struct DCMPositionResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = DCMPositionDataTpl<Scalar>;

  Model pin_model_;

  DCMPositionResidualTpl(const int ndx, const int nu, const Model &model,
                         const Vector3s &dcm_ref, const double alpha)
      : Base(ndx, nu, 3), pin_model_(model), dcm_ref_(dcm_ref), alpha_(alpha) {}

  const Vector3s &getReference() const { return dcm_ref_; }
  void setReference(const Eigen::Ref<const Vector3s> &new_ref) {
    dcm_ref_ = new_ref;
  }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(*this);
  }

protected:
  Vector3s dcm_ref_;
  double alpha_;
};

template <typename Scalar>
struct DCMPositionDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;
  /// Jacobian of the error
  typename math_types<Scalar>::Matrix3Xs fJf_;

  DCMPositionDataTpl(const DCMPositionResidualTpl<Scalar> &model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./dcm-position.txx"
#endif
