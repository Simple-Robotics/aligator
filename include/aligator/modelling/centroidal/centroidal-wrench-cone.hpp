#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

/**
 * @brief This residual implements the wrench cone for a
 * centroidal model with control \f$u = (f_1,...,f_c) \f$
 * with \f$f_k\f$ 6D spatial force.
 *
 * @details Considering an contact k exerting 6D force f,
 * the residual returns \f$ A f \f$ with \f$A \in \mathbb{R}^{17 \times 6}\f$
 * the wrench cone matrix gathering Coulomb friction inequalities,
 * CoP inequalities and limits on vertical torque. The usual
 * wrench cone approximation with 4 facets is leveraged here.
 * The frame in contact is supposed to be rectangular.
 */

template <typename Scalar> struct CentroidalWrenchConeDataTpl;

template <typename _Scalar>
struct CentroidalWrenchConeResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = CentroidalWrenchConeDataTpl<Scalar>;

  CentroidalWrenchConeResidualTpl(const int ndx, const int nu, const int k,
                                  const double mu, const double half_length,
                                  const double half_width)
      : Base(ndx, nu, 17), k_(k), mu_(mu), hL_(half_length), hW_(half_width) {}

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  int k_;     // Contact index corresponding to the contact frame
  double mu_; // Friction coefficient
  double hL_; // Half-length of the contact frame
  double hW_; // Half-width of the contact frame
};

template <typename Scalar>
struct CentroidalWrenchConeDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  Eigen::Matrix<Scalar, 17, 6> Jtemp_;

  CentroidalWrenchConeDataTpl(
      const CentroidalWrenchConeResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./centroidal-wrench-cone.txx"
#endif
