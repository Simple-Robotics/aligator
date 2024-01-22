#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

template <typename Scalar> struct FrictionConeDataTpl;

template <typename _Scalar>
struct FrictionConeResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = FrictionConeDataTpl<Scalar>;

  FrictionConeResidualTpl(const int ndx, const int nu, const int k,
                          const double mu)
      : Base(ndx, nu, 2), nk_(nu / 3), k_(k), mu2_(mu * mu) {
    if (k_ >= nk_) {
      ALIGATOR_RUNTIME_ERROR(
          fmt::format("Invalid contact index: k should be < {}. ", nk_));
    }
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &u,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

protected:
  const std::size_t nk_;
  const std::size_t k_;
  const double mu2_;
};

template <typename Scalar>
struct FrictionConeDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  FrictionConeDataTpl(const FrictionConeResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/centroidal/friction-cone.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./friction-cone.txx"
#endif
