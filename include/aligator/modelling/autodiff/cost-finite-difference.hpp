#pragma once
#include "aligator/core/cost-abstract.hpp"
#include "aligator/core/manifold-base.hpp"

namespace aligator {
namespace autodiff {

template <typename Scalar>
struct CostFiniteDifferenceHelper : CostAbstractTpl<Scalar> {
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostBase = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  using CostBase::space;

  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  struct Data;

  CostFiniteDifferenceHelper(xyz::polymorphic<CostBase> cost,
                             const Scalar fd_eps);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data_) const override;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data_) const override;

  /// @brief Compute the cost Hessians \f$(\ell_{ij})_{i,j \in \{x,u\}}\f$
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &) const override {}

  auto createData() const -> shared_ptr<CostData> override;

  xyz::polymorphic<CostBase> cost_;
  Scalar fd_eps;
};

template <typename Scalar>
struct CostFiniteDifferenceHelper<Scalar>::Data : CostData {

  shared_ptr<CostData> c1, c2;
  VectorXs dx, du;
  VectorXs xp, up;

  Data(CostFiniteDifferenceHelper const &obj)
      : CostData(obj)
      , dx(obj.ndx())
      , du(obj.nu)
      , xp(obj.nx())
      , up(obj.nu) {
    c1 = obj.cost_->createData();
    c2 = obj.cost_->createData();
  }
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct CostFiniteDifferenceHelper<context::Scalar>;
#endif

} // namespace autodiff
} // namespace aligator
