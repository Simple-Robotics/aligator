#pragma once

#include "./fwd.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-jacobian.hpp>

#include <Eigen/LU>

namespace proxddp {

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
    const RigidConstraintModelTpl<Scalar, Options> &constraint_model,
    RigidConstraintDataTpl<Scalar, Options> &constraint_data,
    const Eigen::MatrixBase<OutType> &res_) {
  namespace pin = pinocchio;
  using MatrixXs = Eigen::Matrix<Scalar, -1, -1>;

  OutType &res = res_.const_cast_derived();

  long nu = actMatrix.cols();
  long nv = model.nv;
  assert(nv == actMatrix.rows() && "Actuation matrix dimension inconsistent.");

  pin::computeAllTerms(model, data, q, v);
  const auto &nle = data.nle;

  long d = (long)constraint_model.size();
  assert(res.size() == nu + d);
  MatrixXs work(nv, nu + d);

  work.leftCols(nu) = actMatrix;
  auto JacT = work.rightCols(d);
  pin::getConstraintJacobian(model, data, constraint_model, constraint_data,
                             JacT.transpose());
  JacT = -JacT;

  Eigen::ColPivHouseholderQR<MatrixXs> lu(work);

  res = lu.solve(nle);
}

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./constrained-rnea.txx"
#endif
