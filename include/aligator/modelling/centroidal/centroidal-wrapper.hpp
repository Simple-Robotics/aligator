#pragma once

#include "aligator/core/unary-function.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

template <typename Scalar> struct CentroidalWrapperDataTpl;

/**
 * @brief This residual acts as a wrapper for centroidal model cost
 * functions in which the external forces are added to the state
 * and the control becomes the forces derivatives.
 */

template <typename _Scalar>
struct CentroidalWrapperResidualTpl : UnaryFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using FunPtr = xyz::polymorphic<StageFunctionTpl<Scalar>>;
  using BaseData = typename Base::Data;
  using Data = CentroidalWrapperDataTpl<Scalar>;

  CentroidalWrapperResidualTpl(FunPtr centroidal_cost)
      : Base(centroidal_cost->ndx1 + centroidal_cost->nu, centroidal_cost->nu,
             centroidal_cost->nr)
      , centroidal_cost_(centroidal_cost) {}

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

  FunPtr centroidal_cost_;
};

template <typename Scalar>
struct CentroidalWrapperDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  shared_ptr<Base> wrapped_data_;

  CentroidalWrapperDataTpl(const CentroidalWrapperResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/centroidal/centroidal-wrapper.txx"
#endif
