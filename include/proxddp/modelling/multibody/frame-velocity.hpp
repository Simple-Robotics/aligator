#pragma once

#include "proxddp/core/function-abstract.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

namespace proxddp {

template <typename Scalar> struct FrameVelocityDataTpl;

template <typename _Scalar>
struct FrameVelocityResidualTpl : StageFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Motion = pinocchio::MotionTpl<Scalar>;
  using Data = FrameVelocityDataTpl<Scalar>;

  shared_ptr<Model> pin_model_;
  pinocchio::FrameIndex pin_frame_id_;

  FrameVelocityResidualTpl(const int ndx, const int nu,
                           const shared_ptr<Model> &model,
                           const Motion &velocity,
                           const pinocchio::FrameIndex id,
                           const pinocchio::ReferenceFrame type)
      : Base(ndx, nu, 6), pin_model_(model), pin_frame_id_(id),
        v_ref_(velocity), type_(type) {}

  pinocchio::FrameIndex getFrameId() const { return pin_frame_id_; }
  void setFrameId(const std::size_t id) { pin_frame_id_ = id; }

  const Motion &getReference() const { return v_ref_; }
  void setReference(const Motion &v_new) { v_ref_ = v_new; }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  Motion v_ref_;
  pinocchio::ReferenceFrame type_;
};

template <typename Scalar>
struct FrameVelocityDataTpl : FunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = FunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using Motion = pinocchio::MotionTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;

  FrameVelocityDataTpl(const FrameVelocityResidualTpl<Scalar> *model);
};

} // namespace proxddp

#include "proxddp/modelling/multibody/frame-velocity.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./frame-velocity.txx"
#endif
