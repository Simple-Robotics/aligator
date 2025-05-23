/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/spaces/multibody.hpp"
#include <pinocchio/multibody/data.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct MultibodyFreeFwdDataTpl;

/**
 * @brief   Free-space multibody forward dynamics, using Pinocchio.
 *
 * @details This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$
 * (the phase space \f$x = (q,v)\f$) using the differential equation \f[ \dot{x}
 * = f(x, u) = \begin{bmatrix} v \\ a(q, v, \tau(u)) \end{bmatrix} \f] where
 * \f$\tau(u) = Bu\f$, \f$B\f$ is a given actuation matrix, and
 * \f$a(q,v,\tau)\f$ is the acceleration computed from the ABA algorithm.
 *
 */
template <typename _Scalar>
struct MultibodyFreeFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ContinuousDynamicsDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = MultibodyFreeFwdDataTpl<Scalar>;
  using Manifold = MultibodyPhaseSpace<Scalar>;
  using PolyManifold = xyz::polymorphic<Manifold>;

  using Base::nu_;

  Manifold space_;
  MatrixXs actuation_matrix_;

  const Manifold &space() const { return space_; }
  int ntau() const { return space_.getModel().nv; }

  MultibodyFreeFwdDynamicsTpl(const Manifold &state, const MatrixXs &actuation);
  MultibodyFreeFwdDynamicsTpl(const Manifold &state);

  /**
   * @brief  Determine whether the system is underactuated.
   * @details This is the case when the actuation matrix rank is lower to the
   * velocity dimension.
   */
  bool isUnderactuated() const {
    long nv = space().getModel().nv;
    return act_matrix_rank < nv;
  }

  Eigen::Index getActuationMatrixRank() const { return act_matrix_rank; }

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const;
  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;

private:
  Eigen::FullPivLU<MatrixXs> lu_decomp_;
  Eigen::Index act_matrix_rank;
};

template <typename Scalar>
struct MultibodyFreeFwdDataTpl : ContinuousDynamicsDataTpl<Scalar> {
  using Base = ContinuousDynamicsDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using PinDataType = pinocchio::DataTpl<Scalar>;
  VectorXs tau_;
  MatrixXs dtau_dx_;
  MatrixXs dtau_du_;
  PinDataType pin_data_;
  MultibodyFreeFwdDataTpl(const MultibodyFreeFwdDynamicsTpl<Scalar> *cont_dyn);
};

} // namespace dynamics
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/multibody-free-fwd.txx"
#endif
#endif
