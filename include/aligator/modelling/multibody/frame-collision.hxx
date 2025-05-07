#pragma once

#include "aligator/modelling/multibody/frame-collision.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/collision/distance.hpp>
#include <iostream>

namespace aligator {

template <typename Scalar>
void FrameCollisionResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::forwardKinematics(pin_model_, pdata, x.head(pin_model_.nq));
  pinocchio::updateFramePlacements(pin_model_, pdata);

  // computes the collision distance between pair of frames
  pinocchio::updateGeometryPlacements(pin_model_, pdata, geom_model_,
                                      d.geom_data, x.head(pin_model_.nq));
  pinocchio::computeDistance(geom_model_, d.geom_data, frame_pair_id_);

  // calculate residual
  d.value_[0] = d.geom_data.distanceResults[frame_pair_id_].min_distance;
}

template <typename Scalar>
void FrameCollisionResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                         BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  // Calculate vector from joint to collision p1 and joint to collision p2,
  // expressed in local world aligned
  d.distance_ = d.geom_data.distanceResults[frame_pair_id_].nearest_points[0] -
                pdata.oMf[frame_id1_].translation();
  d.distance2_ = d.geom_data.distanceResults[frame_pair_id_].nearest_points[1] -
                 pdata.oMf[frame_id2_].translation();

  d.jointToP1_.setIdentity();
  d.jointToP1_.translation(d.distance_);

  d.jointToP2_.setIdentity();
  d.jointToP2_.translation(d.distance2_);

  // Get frame Jacobians
  pinocchio::computeJointJacobians(pin_model_, pdata);
  pinocchio::getFrameJacobian(pin_model_, pdata, frame_id1_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.Jcol_);

  pinocchio::getFrameJacobian(pin_model_, pdata, frame_id2_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.Jcol2_);

  // compute Jacobian at p1
  d.Jcol_ = d.jointToP1_.toActionMatrixInverse() * d.Jcol_;

  // compute Jacobian at p2
  d.Jcol2_ = d.jointToP2_.toActionMatrixInverse() * d.Jcol2_;

  // compute the residual derivatives
  d.Jx_.setZero();
  d.Jx_.leftCols(pin_model_.nv) =
      d.geom_data.distanceResults[frame_pair_id_].normal.transpose() *
      (d.Jcol2_.template topRows<3>() - d.Jcol_.template topRows<3>());
}

template <typename Scalar>
FrameCollisionDataTpl<Scalar>::FrameCollisionDataTpl(
    const FrameCollisionResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, 1)
    , pin_data_(model.pin_model_)
    , geom_data(pinocchio::GeometryData(model.geom_model_))
    , Jcol_(6, model.pin_model_.nv)
    , Jcol2_(6, model.pin_model_.nv) {
  Jcol_.setZero();
  Jcol2_.setZero();
}

} // namespace aligator
