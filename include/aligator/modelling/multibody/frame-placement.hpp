#pragma once

#include "aligator/modelling/multibody/fwd.hpp"
#include "aligator/core/unary-function.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

template <typename Scalar> struct FramePlacementDataTpl;

template <typename _Scalar>
struct FramePlacementResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;
  using Model = pinocchio::ModelTpl<Scalar>;
  using PolyManifold = xyz::polymorphic<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FramePlacementDataTpl<Scalar>;

  Model pin_model_;

  FramePlacementResidualTpl(const int ndx, const int nu, const Model &model,
                            const SE3 &frame,
                            const pinocchio::FrameIndex frame_id)
      : Base(ndx, nu, 6)
      , pin_model_(model)
      , p_ref_(frame)
      , p_ref_inverse_(frame.inverse()) {
    pin_frame_id_ = frame_id;
  }

  const SE3 &getReference() const { return p_ref_; }
  void setReference(const SE3 &p_new) {
    p_ref_ = p_new;
    p_ref_inverse_ = p_new.inverse();
  }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }

protected:
  SE3 p_ref_;
  SE3 p_ref_inverse_;
};

template <typename Scalar>
struct FramePlacementDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;
  /// Placement error of the frame.
  SE3 rMf_;
  /// Jacobian of the error
  typename math_types<Scalar>::Matrix6s rJf_;
  /// Jacobian of the error, local frame
  typename math_types<Scalar>::Matrix6Xs fJf_;

  FramePlacementDataTpl(const FramePlacementResidualTpl<Scalar> &model);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct FramePlacementResidualTpl<context::Scalar>;
extern template struct FramePlacementDataTpl<context::Scalar>;
#endif
} // namespace aligator
