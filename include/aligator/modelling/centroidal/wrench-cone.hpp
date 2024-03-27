#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

/**
 * @brief This residual implements the "ice cream" friction cone for a
 * centroidal model with state \f$x = (c, h, L) \f$.
 *
 * @details Considering an unilateral contact k exerting 3D force u,
 * the residual returns a two-dimension array with first component
 * equal to \f$ \epsilon - u_z \f$ (strictly positive normal force
 * condition) and second component equal to * \f$ u_{x,y}^2 -
 * \mu^2 * u_{z}^2 \f$ (non-slippage condition) with \f$ \epsilon
 * \f$ small threshold and \f$ \mu \f$ friction coefficient.
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
                        const double mu, const double L, const double W)
      : Base(ndx, nu, 16), k_(k), mu_(mu), L_(L), W_(W) {}

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

protected:
  int k_;
  double mu_;
  double L_;
  double W_;
};

template <typename Scalar>
struct WrenchConeDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using Matrix6s = Eigen::Matrix<Scalar, 16, 6>;

  Matrix6s Jtemp_;

  WrenchConeDataTpl(const WrenchConeResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/centroidal/wrench-cone.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./wrench-cone.txx"
#endif
