#pragma once

#include "aligator/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fcl.hpp>
#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>

namespace aligator {

template <typename Scalar> struct FrameCollisionDataTpl;

template <typename _Scalar>
struct FrameCollisionResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = xyz::polymorphic<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FrameCollisionDataTpl<Scalar>;
  using GeometryModel = pinocchio::GeometryModel;

  Model pin_model_;
  GeometryModel geom_model_;

  FrameCollisionResidualTpl(const int ndx, const int nu, const Model &model,
                            const GeometryModel &geom_model,
                            const pinocchio::PairIndex frame_pair_id,
                            const double alpha)
      : Base(ndx, nu, 1), pin_model_(model), geom_model_(geom_model),
        frame_pair_id_(frame_pair_id), alpha_(alpha) {
    frame_id1_ =
        geom_model
            .geometryObjects[geom_model.collisionPairs[frame_pair_id_].first]
            .parentFrame;
    frame_id2_ =
        geom_model
            .geometryObjects[geom_model.collisionPairs[frame_pair_id_].second]
            .parentFrame;
  }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(*this);
  }

protected:
  pinocchio::PairIndex frame_pair_id_;
  double alpha_;
  pinocchio::FrameIndex frame_id1_;
  pinocchio::FrameIndex frame_id2_;
};

template <typename Scalar>
struct FrameCollisionDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using PinGeom = pinocchio::GeometryData;

  /// Pinocchio data object
  PinData pin_data_;
  /// Pinocchio geometry object
  PinGeom geometry_;
  /// Jacobian of the collision point
  typename math_types<Scalar>::Matrix6Xs Jcol_;
  typename math_types<Scalar>::Matrix6Xs Jcol2_;
  /// Distance between witness points
  typename math_types<Scalar>::Vector3s witness_distance_;
  /// Distance from nearest point to joint for each collision frame
  typename math_types<Scalar>::Vector3s distance_;
  typename math_types<Scalar>::Vector3s distance2_;
  /// Norm of the witness distance
  double witness_norm_;

  FrameCollisionDataTpl(const FrameCollisionResidualTpl<Scalar> &model);
};

} // namespace aligator

#include "aligator/modelling/multibody/frame-collision.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./frame-collision.txx"
#endif
