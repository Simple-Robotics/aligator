#pragma once

#include "aligator/modelling/multibody/frame-collision.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/collision/distance.hpp>

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
                                      d.geometry_, x.head(pin_model_.nq));
  pinocchio::computeDistance(geom_model_, d.geometry_, frame_pair_id_);

  // calculate residual
  d.witness_distance_ =
      d.geometry_.distanceResults[frame_pair_id_].nearest_points[0] -
      d.geometry_.distanceResults[frame_pair_id_].nearest_points[1];

  d.witness_norm_ = d.witness_distance_.norm();
  if (d.witness_norm_ < alpha_) {
    d.value_[0] =
        Scalar(0.5) * (d.witness_norm_ - alpha_) * (d.witness_norm_ - alpha_);
  } else {
    d.value_[0] = Scalar(0.0);
  }
  // d.value_[0] = -d.witness_distance_.norm() + alpha_;
  // std::cout << "value " << d.value_[0] << std::endl;
}

template <typename Scalar>
void FrameCollisionResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                         BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  // calculate vector from joint to collision p1 and joint to collision p2,
  // expressed in local world aligned
  d.distance_ = d.geometry_.distanceResults[frame_pair_id_].nearest_points[0] -
                pdata.oMf[frame_id1_].translation();
  d.distance2_ = d.geometry_.distanceResults[frame_pair_id_].nearest_points[1] -
                 pdata.oMf[frame_id2_].translation();
  pinocchio::computeJointJacobians(pin_model_, pdata);

  pinocchio::getFrameJacobian(pin_model_, pdata, frame_id1_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.Jcol_);

  pinocchio::getFrameJacobian(pin_model_, pdata, frame_id2_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.Jcol2_);

  // compute Jacobian at p1
  d.Jcol_.template topRows<3>().noalias() +=
      pinocchio::skew(d.distance_).transpose() *
      d.Jcol_.template bottomRows<3>();

  // compute Jacobian at p2
  d.Jcol2_.template topRows<3>().noalias() +=
      pinocchio::skew(d.distance2_).transpose() *
      d.Jcol2_.template bottomRows<3>();

  Eigen::MatrixXd J =
      d.Jcol_.template topRows<3>() - d.Jcol2_.template topRows<3>();

  // compute the residual derivatives
  d.Jx_.setZero();
  if (d.witness_norm_ < alpha_ and d.witness_norm_ > 0) {
    d.Jx_.leftCols(pin_model_.nv) = d.witness_distance_.transpose() *
                                    (d.witness_norm_ - alpha_) /
                                    d.witness_norm_ * J;
  }
}

template <typename Scalar>
FrameCollisionDataTpl<Scalar>::FrameCollisionDataTpl(
    const FrameCollisionResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, 1), pin_data_(model.pin_model_),
      geometry_(pinocchio::GeometryData(model.geom_model_)),
      Jcol_(6, model.pin_model_.nv), Jcol2_(6, model.pin_model_.nv) {
  Jcol_.setZero();
  Jcol2_.setZero();
  witness_norm_ = 0;
}

} // namespace aligator
