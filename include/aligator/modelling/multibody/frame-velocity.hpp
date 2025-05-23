#pragma once

#include "aligator/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/frame.hpp>
#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

template <typename Scalar> struct FrameVelocityDataTpl;

template <typename _Scalar>
struct FrameVelocityResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = xyz::polymorphic<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Motion = pinocchio::MotionTpl<Scalar>;
  using Data = FrameVelocityDataTpl<Scalar>;

  Model pin_model_;

  FrameVelocityResidualTpl(const int ndx, const int nu, const Model &model,
                           const Motion &velocity,
                           const pinocchio::FrameIndex id,
                           const pinocchio::ReferenceFrame type);

  const Motion &getReference() const { return vref_; }
  void setReference(const Motion &v_new) { vref_ = v_new; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;
  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }

protected:
  Motion vref_;
  pinocchio::ReferenceFrame type_;
};

template <typename Scalar>
struct FrameVelocityDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using Motion = pinocchio::MotionTpl<Scalar>;

  /// Pinocchio data object.
  pinocchio::DataTpl<Scalar> pin_data_;

  FrameVelocityDataTpl(const FrameVelocityResidualTpl<Scalar> &model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/multibody/frame-velocity.txx"
#endif
