#pragma once

#include "aligator/modelling/multibody/fwd.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-jacobian.hpp>

#include <Eigen/LU>

namespace aligator {

namespace {
using pinocchio::DataTpl;
using pinocchio::ModelTpl;
using pinocchio::RigidConstraintDataTpl;
using pinocchio::RigidConstraintModelTpl;
} // namespace

template <typename Scalar, typename ConfigType, typename VelType,
          typename MatrixType, typename OutType, int Options>
void underactuatedConstrainedInverseDynamics(
    const ModelTpl<Scalar, Options> &model, DataTpl<Scalar, Options> &data,
    const Eigen::MatrixBase<ConfigType> &q, Eigen::MatrixBase<VelType> const &v,
    const Eigen::MatrixBase<MatrixType> &actMatrix,
    const pinocchio::container::aligned_vector<
        RigidConstraintModelTpl<Scalar, Options>> &constraint_models,
    pinocchio::container::aligned_vector<
        RigidConstraintDataTpl<Scalar, Options>> &constraint_datas,
    const Eigen::MatrixBase<OutType> &res_) {
  namespace pin = pinocchio;
  using MatrixXs = Eigen::Matrix<Scalar, -1, -1>;
  assert(constraint_models.size() == constraint_datas.size() &&
         "constraint_models and constraint_datas do not have the same size");

  OutType &res = res_.const_cast_derived();

  long nu = actMatrix.cols();
  long nv = model.nv;
  assert(nv == actMatrix.rows() && "Actuation matrix dimension inconsistent.");

  pin::computeAllTerms(model, data, q, v);
  const auto &nle = data.nle;

  int d = 0;
  for (size_t k = 0; k < constraint_models.size(); ++k) {
    d += (int)constraint_models[k].size();
  }

  assert(res.size() == nu + d);
  MatrixXs work(nv, nu + d);

  work.leftCols(nu) = actMatrix;
  auto JacT = work.rightCols(d);
  pin::getConstraintsJacobian(model, data, constraint_models, constraint_datas,
                              JacT.transpose());
  JacT = -JacT;

  Eigen::ColPivHouseholderQR<MatrixXs> qr(work);

  res = qr.solve(nle);
}

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template void underactuatedConstrainedInverseDynamics<
    context::Scalar, context::ConstVectorRef, context::ConstVectorRef,
    context::ConstMatrixRef, context::VectorRef, context::Options>(
    const context::PinModel &, context::PinData &,
    const Eigen::MatrixBase<context::ConstVectorRef> &,
    const Eigen::MatrixBase<context::ConstVectorRef> &,
    const Eigen::MatrixBase<context::ConstMatrixRef> &,
    const PINOCCHIO_ALIGNED_STD_VECTOR(context::RCM) &,
    PINOCCHIO_ALIGNED_STD_VECTOR(context::RCD) &,
    const Eigen::MatrixBase<context::VectorRef> &);
#endif
} // namespace aligator
