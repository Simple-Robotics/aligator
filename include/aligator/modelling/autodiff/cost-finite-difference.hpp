#pragma once
#include "aligator/core/cost-abstract.hpp"
#include <proxsuite-nlp/manifold-base.hpp>

namespace aligator {
namespace autodiff {

template <typename Scalar>
struct CostFiniteDifferenceHelper : CostAbstractTpl<Scalar> {
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostBase = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  using CostBase::space;

  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  struct Data : CostData {

    shared_ptr<CostData> c1, c2;
    VectorXs dx, du;
    VectorXs xp, up;

    Data(CostFiniteDifferenceHelper const &obj)
        : CostData(obj), dx(obj.ndx()), du(obj.nu), xp(obj.nx()), up(obj.nu) {
      c1 = obj.cost_->createData();
      c2 = obj.cost_->createData();
    }
  };

  CostFiniteDifferenceHelper(shared_ptr<CostBase> cost, const Scalar fd_eps)

      : CostBase(cost->space, cost->nu), cost_(cost), fd_eps(fd_eps) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data_) const override {
    Data &d = static_cast<Data &>(data_);
    cost_->evaluate(x, u, *d.c1);

    d.value_ = d.c1->value_;
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data_) const override {
    Data &d = static_cast<Data &>(data_);
    Manifold const &space = *this->space;

    cost_->evaluate(x, u, *d.c1);

    d.dx.setZero();
    for (int i = 0; i < this->ndx(); i++) {
      d.dx[i] = fd_eps;
      space.integrate(x, d.dx, d.xp);
      cost_->evaluate(d.xp, u, *d.c2);

      d.Lx_[i] = (d.c2->value_ - d.c1->value_) / fd_eps;
      d.dx[i] = 0.;
    }

    d.du.setZero();
    for (int i = 0; i < this->nu; i++) {
      d.du[i] = fd_eps;
      d.up = u + d.du;
      cost_->evaluate(x, d.up, *d.c2);

      d.Lu_[i] = (d.c2->value_ - d.c1->value_) / fd_eps;
      d.du[i] = 0.;
    }
  }

  /// @brief Compute the cost Hessians \f$(\ell_{ij})_{i,j \in \{x,u\}}\f$
  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &) const override {}

  shared_ptr<CostData> createData() const override {
    return std::make_shared<Data>(*this);
  }

  shared_ptr<CostBase> cost_;
  Scalar fd_eps;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct CostFiniteDifferenceHelper<context::Scalar>;
#endif

} // namespace autodiff
} // namespace aligator
