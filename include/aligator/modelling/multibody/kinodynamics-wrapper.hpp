#pragma once

#include "aligator/core/unary-function.hpp"
#include "aligator/core/function-abstract.hpp"

namespace aligator {

template <typename Scalar> struct KinodynamicsWrapperDataTpl;

/**
 * @brief This residual acts as a wrapper for kinematics model cost
 * functions in which the state is the concatenation of centroidal
 * momentum and multibody joint positions / velocities.
 */

template <typename _Scalar>
struct KinodynamicsWrapperResidualTpl : UnaryFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using FunPtr = shared_ptr<UnaryFunctionTpl<Scalar>>;
  using BaseData = typename Base::Data;
  using Data = KinodynamicsWrapperDataTpl<Scalar>;

  KinodynamicsWrapperResidualTpl(FunPtr multibody_cost, const int &nq,
                                 const int &nv, const int &nk)
      : Base(multibody_cost->ndx1, nv - 6 + 3 * nk, multibody_cost->nr),
        multibody_cost_(multibody_cost), nx_(nq + nv), nk_(nk) {}

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

  FunPtr multibody_cost_;
  const int nx_;
  const int nk_;
};

template <typename Scalar>
struct KinodynamicsWrapperDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  shared_ptr<Base> wrapped_data_;

  KinodynamicsWrapperDataTpl(
      const KinodynamicsWrapperResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/multibody/kinodynamics-wrapper.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./kinodynamics-wrapper.txx"
#endif
