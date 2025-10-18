#pragma once

#include "aligator/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>

namespace aligator {

template <typename Scalar> struct CenterOfMassVelocityDataTpl;

template <typename _Scalar>
struct CenterOfMassVelocityResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = CenterOfMassVelocityDataTpl<Scalar>;

  Model pin_model_;
  Vector3s v_ref_;

  CenterOfMassVelocityResidualTpl(const int ndx, const int nu,
                                  const Model &model, const Vector3s &frame_vel)
      : Base(ndx, nu, 3)
      , pin_model_(model)
      , v_ref_(frame_vel) {}

  ALIGATOR_DEPRECATED const Vector3s &getReference() const { return v_ref_; }
  ALIGATOR_DEPRECATED void
  setReference(const Eigen::Ref<const Vector3s> &v_new) {
    v_ref_ = v_new;
  }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }
};

template <typename Scalar>
struct CenterOfMassVelocityDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;
  /// Jacobian of the error
  typename math_types<Scalar>::Matrix3Xs fJf_;

  CenterOfMassVelocityDataTpl(
      const CenterOfMassVelocityResidualTpl<Scalar> &model);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct CenterOfMassVelocityResidualTpl<context::Scalar>;
extern template struct CenterOfMassVelocityDataTpl<context::Scalar>;
#endif
} // namespace aligator
