/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/centroidal-kinematics-fwd.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
CentroidalKinematicsFwdDynamicsTpl<Scalar>::CentroidalKinematicsFwdDynamicsTpl(
    const ManifoldPtr &state, const int &nv, CentroidalPtr centroidal)
    : Base(state, nv + (int)centroidal->nu_), space_(state),
      nuc_(centroidal->nu_), nv_(nv), centroidal_(centroidal) {}

template <typename Scalar>
void CentroidalKinematicsFwdDynamicsTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  centroidal_->forward(x.head(9), u.head(nuc_), *d.centroidal_data_);

  d.xdot_.template head<9>() = d.centroidal_data_->xdot_;
  d.xdot_.segment(9, nv_) = x.tail(nv_);
  d.xdot_.tail(nv_) = u.tail(nv_);
}

template <typename Scalar>
void CentroidalKinematicsFwdDynamicsTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &u, BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  centroidal_->dForward(x.head(9), u.head(nuc_), *d.centroidal_data_);

  d.Jx_.template topLeftCorner<9, 9>() = d.centroidal_data_->Jx_;
  d.Ju_.topLeftCorner(9, nuc_) = d.centroidal_data_->Ju_;
}

template <typename Scalar>
shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
CentroidalKinematicsFwdDynamicsTpl<Scalar>::createData() const {
  return allocate_shared_eigen_aligned<Data>(this);
}

template <typename Scalar>
CentroidalKinematicsFwdDataTpl<Scalar>::CentroidalKinematicsFwdDataTpl(
    const CentroidalKinematicsFwdDynamicsTpl<Scalar> *model)
    : Base(model->ndx(), model->nu()) {
  centroidal_data_ =
      std::dynamic_pointer_cast<Base>(model->centroidal_->createData());
  this->Jx_.block(9, 9 + model->nv_, model->nv_, model->nv_).setIdentity();
  this->Ju_.bottomRightCorner(model->nv_, model->nv_).setIdentity();
}
} // namespace dynamics
} // namespace aligator
