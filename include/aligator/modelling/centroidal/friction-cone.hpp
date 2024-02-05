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

  FrictionConeResidualTpl(const std::size_t &ndx, const std::size_t &nu,
                          const std::size_t &k, const double &mu,
                          const double &epsilon)
      : Base(ndx, nu, 2), nk_(nu / 3), k_(k), mu2_(mu * mu), epsilon_(epsilon) {
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
  const double epsilon_;
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
