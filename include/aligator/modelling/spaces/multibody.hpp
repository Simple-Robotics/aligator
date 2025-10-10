/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include "aligator/modelling/spaces/tangent-bundle.hpp"

namespace aligator {

/** @brief    Multibody configuration group \f$\mathcal{Q}\f$, defined using the
 * Pinocchio library.
 *
 *  @details  This uses a pinocchio::ModelTpl object to define the manifold.
 */
template <typename _Scalar>
struct MultibodyConfiguration : ManifoldAbstractTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Self = MultibodyConfiguration<Scalar>;
  using ModelType = pinocchio::ModelTpl<Scalar>;
  using Base = ManifoldAbstractTpl<Scalar>;
  using Base::ndx;
  using Base::nx;

  MultibodyConfiguration(const ModelType &model)
      : Base(model.nq, model.nv)
      , model_(model) {};
  MultibodyConfiguration(const MultibodyConfiguration &) = default;
  MultibodyConfiguration &operator=(const MultibodyConfiguration &) = default;
  MultibodyConfiguration(MultibodyConfiguration &&) = default;
  MultibodyConfiguration &operator=(MultibodyConfiguration &&) = default;

  const ModelType &getModel() const { return model_; }

  bool isNormalized(const ConstVectorRef &x) const {
    return pinocchio::isNormalized(model_, x);
  }

protected:
  ModelType model_;

  /// \name implementations
  /// \{

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef xout) const {
    pinocchio::integrate(model_, x, v, xout);
  }

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      pinocchio::dIntegrate(model_, x, v, Jout, pinocchio::ARG0);
      break;
    case 1:
      pinocchio::dIntegrate(model_, x, v, Jout, pinocchio::ARG1);
      break;
    }
  }

  void JintegrateTransport_impl(const ConstVectorRef &x,
                                const ConstVectorRef &v, MatrixRef Jout,
                                int arg) const {
    switch (arg) {
    case 0:
      pinocchio::dIntegrateTransport(model_, x, v, Jout, pinocchio::ARG0);
      break;
    case 1:
      pinocchio::dIntegrateTransport(model_, x, v, Jout, pinocchio::ARG1);
      break;
    default:
      break;
    }
  }

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef vout) const {
    pinocchio::difference(model_, x0, x1, vout);
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      pinocchio::dDifference(model_, x0, x1, Jout, pinocchio::ARG0);
      break;
    case 1:
      pinocchio::dDifference(model_, x0, x1, Jout, pinocchio::ARG1);
      break;
    }
  }

  void interpolate_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        const Scalar &u, VectorRef out) const {
    pinocchio::interpolate(model_, x0, x1, u, out);
  }

  void neutral_impl(VectorRef out) const { pinocchio::neutral(model_, out); }

  void rand_impl(VectorRef out) const {
    pinocchio::randomConfiguration(model_, model_.lowerPositionLimit,
                                   model_.upperPositionLimit, out);
  }

  /// \}
};

/** @brief      The tangent bundle of a multibody configuration group.
 *  @details    This is not a typedef, since we provide a constructor for the
 * class. Any point on the manifold is of the form \f$x = (q,v) \f$, where \f$q
 * \in \mathcal{Q} \f$ is a configuration and \f$v\f$ is a joint velocity
 * vector.
 */
template <typename Scalar>
struct MultibodyPhaseSpace : TangentBundleTpl<MultibodyConfiguration<Scalar>> {
  using ConfigSpace = MultibodyConfiguration<Scalar>;
  using ModelType = typename ConfigSpace::ModelType;

  const ModelType &getModel() const { return this->base_.getModel(); }

  MultibodyPhaseSpace(const ModelType &model)
      : TangentBundleTpl<ConfigSpace>(ConfigSpace(model)) {}
  MultibodyPhaseSpace(const MultibodyPhaseSpace &) = default;
  MultibodyPhaseSpace &operator=(const MultibodyPhaseSpace &) = default;
  MultibodyPhaseSpace(MultibodyPhaseSpace &&) = default;
  MultibodyPhaseSpace &operator=(MultibodyPhaseSpace &&) = default;
};

} // namespace aligator
