/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/dynamics/multibody-common.hpp"
#include "aligator/core/common-model-builder-container.hpp"

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

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
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = MultibodyFreeFwdDataTpl<Scalar>;
  using Manifold = proxsuite::nlp::MultibodyPhaseSpace<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using MultibodyCommon = MultibodyCommonTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  using Base::nu_;

  ManifoldPtr space_;
  MatrixXs actuation_matrix_;

  const Manifold &space() const { return *space_; }
  int ntau() const { return space_->getModel().nv; }

  MultibodyFreeFwdDynamicsTpl(const ManifoldPtr &state,
                              const MatrixXs &actuation);
  MultibodyFreeFwdDynamicsTpl(const ManifoldPtr &state);

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

  void configure(CommonModelBuilderContainer &container) const override;
  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const override;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override;

  shared_ptr<ContDataAbstract>
  createData(const CommonModelDataContainer &container) const override;
  shared_ptr<ContDataAbstract> createData() const override {
    ALIGATOR_RUNTIME_ERROR("createData can't be called without arguments");
  }

private:
  Eigen::FullPivLU<MatrixXs> lu_decomp_;
  Eigen::Index act_matrix_rank;
};

template <typename Scalar> struct MultibodyFreeFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using MultibodyCommon = MultibodyCommonTpl<Scalar>;
  using MultibodyCommonData = MultibodyCommonDataTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  MultibodyFreeFwdDataTpl(const MultibodyFreeFwdDynamicsTpl<Scalar> *cont_dyn,
                          const CommonModelDataContainer &container);

  const MultibodyCommonData *multibody_data_;
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/multibody-free-fwd.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/dynamics/multibody-free-fwd.txx"
#endif
