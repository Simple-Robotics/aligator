/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "proxddp/core/function-abstract.hpp"
#include "proxddp/core/cost-abstract.hpp"
#include <proxnlp/manifold-base.hpp>

namespace proxddp {
namespace autodiff {

enum FDLevel {
  TOC1 = 0, ///< Cast to a \f$C^1\f$ function.
  TOC2 = 1  ///< Cast to a \f$C^2\f$ function.
};

namespace internal {

// fwd declare the implementation of finite difference algorithms.
template <typename _Scalar, FDLevel n> struct finite_difference_impl;

template <typename _Scalar>
struct finite_difference_impl<_Scalar, TOC1>
    : virtual StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = FunctionDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  shared_ptr<Manifold> space_;
  shared_ptr<Base> func_;
  Scalar fd_eps;
  int nx1, nx2;

  struct Data : BaseData {
    using BaseData::ndx1;
    using BaseData::ndx2;
    using BaseData::nr;
    using BaseData::nu;
    shared_ptr<BaseData> data_0;
    shared_ptr<BaseData> data_1;
    VectorXs dx, du, dy;
    VectorXs xp, up, yp;

    Data(finite_difference_impl const &model)
        : BaseData(model.ndx1, model.nu, model.ndx2, model.nr),
          data_0(model.func_->createData()), data_1(model.func_->createData()),
          dx(ndx1), du(nu), dy(ndx2), xp(model.nx1), up(model.nu),
          yp(model.nx2) {
      dx.setZero();
      du.setZero();
      dy.setZero();
    }
  };

  finite_difference_impl(shared_ptr<Manifold> space, shared_ptr<Base> func,
                         const Scalar fd_eps)
      : Base(func->ndx1, func->nu, func->ndx2, func->nr), space_(space),
        func_(func), fd_eps(fd_eps), nx1(space->nx()), nx2(space->nx()) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const {
    Data &d = static_cast<Data &>(data);
    func_->evaluate(x, u, y, *d.data_0);
    d.value_ = d.data_0->value_;
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const {
    Data &d = static_cast<Data &>(data);

    VectorXs &v0 = d.data_0->value_;
    VectorXs &vp = d.data_1->value_;

    for (int i = 0; i < func_->ndx1; i++) {
      d.dx[i] = fd_eps;
      space_->integrate(x, d.dx, d.xp);
      func_->evaluate(d.xp, u, y, *d.data_1);
      data.Jx_.col(i) = (vp - v0) / fd_eps;
      d.dx[i] = 0.;
    }

    for (int i = 0; i < func_->ndx2; i++) {
      d.dy[i] = fd_eps;
      space_->integrate(y, d.dy, d.yp);
      func_->evaluate(x, u, d.yp, *d.data_1);
      data.Jy_.col(i) = (vp - v0) / fd_eps;
      d.dy[i] = 0.;
    }

    for (int i = 0; i < func_->nu; i++) {
      d.du[i] = fd_eps;
      d.up = u + d.du;
      func_->evaluate(x, d.up, y, *d.data_1);
      data.Ju_.col(i) = (vp - v0) / fd_eps;
      d.du[i] = 0.;
    }
  }

  void computeVectorHessianProducts(const ConstVectorRef &,
                                    const ConstVectorRef &,
                                    const ConstVectorRef &,
                                    const ConstVectorRef &, BaseData &) const {}

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }
};

} // namespace internal

template <typename _Scalar, FDLevel n = TOC1> struct finite_difference_wrapper;

/** @brief    Approximate the derivatives of a given function
 * using finite differences, to downcast the function to a
 * StageFunctionTpl.
 */
template <typename _Scalar>
struct finite_difference_wrapper<_Scalar, TOC1>
    : internal::finite_difference_impl<_Scalar, TOC1> {
  using Scalar = _Scalar;

  using StageFunction = StageFunctionTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Base = internal::finite_difference_impl<Scalar, TOC1>;
  using Base::computeJacobians;
  using Base::computeVectorHessianProducts;
  using Base::evaluate;

  PROXNLP_DYNAMIC_TYPEDEFS(_Scalar);

  finite_difference_wrapper(shared_ptr<Manifold> space,
                            shared_ptr<StageFunction> func, const Scalar fd_eps)
      : StageFunction(func->ndx1, func->nu, func->ndx2, func->nr),
        Base(space, func, fd_eps) {}
};

template <typename Scalar>
struct cost_finite_difference_wrapper : CostAbstractTpl<Scalar> {
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostBase = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  using CostBase::space;

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  struct Data : CostData {

    shared_ptr<CostData> c1, c2;
    VectorXs dx, du;
    VectorXs xp, up;

    Data(cost_finite_difference_wrapper const &obj)
        : CostData(obj), dx(obj.ndx()), du(obj.nu), xp(obj.nx()), up(obj.nu) {
      c1 = obj.cost_->createData();
      c2 = obj.cost_->createData();
    }
  };

  cost_finite_difference_wrapper(shared_ptr<CostBase> cost, const Scalar fd_eps)

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
  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data_) const override {
    Data &data = static_cast<Data &>(data_);
  }

  shared_ptr<CostData> createData() const override {
    return std::make_shared<Data>(*this);
  }

  shared_ptr<CostBase> cost_;
  Scalar fd_eps;
};

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
extern template struct finite_difference_wrapper<context::Scalar>;
extern template struct cost_finite_difference_wrapper<context::Scalar>;
#endif

} // namespace autodiff
} // namespace proxddp
