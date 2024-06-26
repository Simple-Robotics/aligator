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

template <typename Scalar> struct WrenchConeDataTpl;

template <typename _Scalar>
struct WrenchConeResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = WrenchConeDataTpl<Scalar>;

  WrenchConeResidualTpl(const int ndx, const int nu, const int k,
                        const double mu, const double half_length,
                        const double half_width)
      : Base(ndx, nu, 17), k_(k), mu_(mu), hL_(half_length), hW_(half_width) {}

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

protected:
  int k_;     // Contact index corresponding to the contact frame
  double mu_; // Friction coefficient
  double hL_; // Half-length of the contact frame
  double hW_; // Half-width of the contact frame
};

template <typename Scalar>
struct WrenchConeDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  Eigen::Matrix<Scalar, 17, 6> Jtemp_;

  WrenchConeDataTpl(const WrenchConeResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/centroidal/wrench-cone.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./wrench-cone.txx"
#endif
