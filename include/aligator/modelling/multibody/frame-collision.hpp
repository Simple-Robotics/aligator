#pragma once

#include "aligator/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/algorithm/geometry.hpp>

#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

template <typename Scalar> struct FrameCollisionDataTpl;

template <typename _Scalar>
struct FrameCollisionResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FrameCollisionDataTpl<Scalar>;
  using GeometryModel = pinocchio::GeometryModel;

  Model pin_model_;
  GeometryModel geom_model_;

  FrameCollisionResidualTpl(const int ndx, const int nu, const Model &model,
                            const GeometryModel &geom_model,
                            const pinocchio::PairIndex frame_pair_id)
      : Base(ndx, nu, 1)
      , pin_model_(model)
      , geom_model_(geom_model)
      , frame_pair_id_(frame_pair_id) {
    if (frame_pair_id >= geom_model_.collisionPairs.size()) {
      ALIGATOR_OUT_OF_RANGE_ERROR(
          "Provided collision pair index {:d} is not valid "
          "(geom model has {:d} pairs).",
          frame_pair_id, geom_model.collisionPairs.size());
    }
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
    return std::make_shared<Data>(*this);
  }

protected:
  pinocchio::PairIndex frame_pair_id_;
  pinocchio::FrameIndex frame_id1_;
  pinocchio::FrameIndex frame_id2_;
};

template <typename Scalar>
struct FrameCollisionDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using typename Base::Matrix6Xs;
  using typename Base::Vector3s;
  using SE3 = pinocchio::SE3Tpl<Scalar>;

  pinocchio::DataTpl<Scalar> pin_data_;
  pinocchio::GeometryData geom_data;
  /// Jacobian of the collision point
  Matrix6Xs Jcol_;
  Matrix6Xs Jcol2_;
  /// Placement of collision point to joint
  SE3 jointToP1_;
  SE3 jointToP2_;
  /// Distance from nearest point to joint for each collision frame
  Vector3s distance_;
  Vector3s distance2_;

  FrameCollisionDataTpl(const FrameCollisionResidualTpl<Scalar> &model);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct FrameCollisionResidualTpl<context::Scalar>;
extern template struct FrameCollisionDataTpl<context::Scalar>;
#endif
} // namespace aligator
