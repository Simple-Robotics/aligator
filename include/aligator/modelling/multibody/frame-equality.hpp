#pragma once

#include "aligator/modelling/multibody/fwd.hpp"
#include "aligator/core/unary-function.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

template <typename Scalar> struct FrameEqualityDataTpl;

/**
 * @brief Residual enforcing equality between two Pinocchio frames.
 *
 * Computes the 6D log error between frame `frame_id1` and frame `frame_id2`.
 * The residual is:
 *
 *     r = log6( f1^{-1} * f1Mf2_ref * f2 )
 *
 * where:
 *   - f1 and f2 are the world placements of the two frames,
 *   - f1Mf2_ref is the desired relative transform (reference).
 *
 * All computations (residual and Jacobians) are expressed in an R frame
 * attached to f1 (f1 -> f1Mf2_ref -> R -> error -> f2)
 */
template <typename _Scalar>
struct FrameEqualityResidualTpl : UnaryFunctionTpl<_Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;
  using Model = pinocchio::ModelTpl<Scalar>;
  using PolyManifold = xyz::polymorphic<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FrameEqualityDataTpl<Scalar>;

  Model pin_model_;

  FrameEqualityResidualTpl(const int ndx, const int nu, const Model &model,
                           const pinocchio::FrameIndex frame_id1,
                           const pinocchio::FrameIndex frame_id2,
                           const SE3 f1Mf2_ref = SE3::Identity())
      : Base(ndx, nu, 6)
      , pin_model_(model)
      , pin_frame_id1_(frame_id1)
      , pin_frame_id2_(frame_id2)
      , f1MR_ref_(f1Mf2_ref) {}

  // Getters and setters
  pinocchio::FrameIndex getFrame1Id() const { return pin_frame_id1_; }
  void setFrame1Id(const std::size_t id) { pin_frame_id1_ = id; }
  pinocchio::FrameIndex getFrame2Id() const { return pin_frame_id2_; }
  void setFrame2Id(const std::size_t id) { pin_frame_id2_ = id; }

  const SE3 &getReference() const { return f1MR_ref_; }
  void setReference(const SE3 &f1Mf2_ref) { f1MR_ref_ = f1Mf2_ref; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }

protected:
  pinocchio::FrameIndex pin_frame_id1_;
  pinocchio::FrameIndex pin_frame_id2_;
  SE3 f1MR_ref_;
};

template <typename Scalar>
struct FrameEqualityDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;
  /// Equality error between the frames.
  SE3 RMf2_;
  /// Jacobian of the error (log6)
  typename math_types<Scalar>::Matrix6s RJlog6f2_;
  /// Jacobian of frame 1 expressed in WORLD
  typename math_types<Scalar>::Matrix6Xs wJf1_;
  /// Jacobian of frame 2 expressed in WORLD
  typename math_types<Scalar>::Matrix6Xs wJf2_;

  FrameEqualityDataTpl(const FrameEqualityResidualTpl<Scalar> &model);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct FrameEqualityResidualTpl<context::Scalar>;
extern template struct FrameEqualityDataTpl<context::Scalar>;
#endif
} // namespace aligator
