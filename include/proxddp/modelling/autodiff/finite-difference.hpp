/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/cost-abstract.hpp"
#include <proxsuite-nlp/manifold-base.hpp>
#include <boost/mpl/bool.hpp>

namespace aligator {
namespace autodiff {

namespace internal {

// fwd declare the implementation of finite difference algorithms.
template <typename _Scalar, template <typename> class _Base>
struct finite_difference_impl : virtual _Base<_Scalar> {
  using Scalar = _Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = _Base<Scalar>;
  using BaseData = typename Base::Data;
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

  template <typename U = Base, class = std::enable_if_t<std::is_same<
                                   U, StageFunctionTpl<Scalar>>::value>>
  finite_difference_impl(shared_ptr<Manifold> space, shared_ptr<U> func,
                         const Scalar fd_eps)
      : Base(func->ndx1, func->nu, func->ndx2, func->nr), space_(space),
        func_(func), fd_eps(fd_eps), nx1(space->nx()), nx2(space->nx()) {}

  template <typename U = Base, class = std::enable_if_t<std::is_same<
                                   U, DynamicsModelTpl<Scalar>>::value>>
  finite_difference_impl(shared_ptr<Manifold> space, shared_ptr<U> func,
                         const Scalar fd_eps, boost::mpl::false_ = {})
      : Base(space, func->nu, space), space_(space), func_(func),
        fd_eps(fd_eps), nx1(space->nx()), nx2(space->nx()) {}

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

template <typename _Scalar> struct FiniteDifferenceHelper;

/** @brief    Approximate the derivatives of a given function
 * using finite differences, to downcast the function to a
 * StageFunctionTpl.
 */
template <typename _Scalar>
struct FiniteDifferenceHelper
    : internal::finite_difference_impl<_Scalar, StageFunctionTpl> {
  using Scalar = _Scalar;

  using StageFunction = StageFunctionTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Base = internal::finite_difference_impl<Scalar, StageFunctionTpl>;
  using Base::computeJacobians;
  using Base::computeVectorHessianProducts;
  using Base::evaluate;

  PROXDDP_DYNAMIC_TYPEDEFS(_Scalar);

  FiniteDifferenceHelper(shared_ptr<Manifold> space,
                         shared_ptr<StageFunction> func, const Scalar fd_eps)
      : StageFunction(func->ndx1, func->nu, func->ndx2, func->nr),
        Base(space, func, fd_eps) {}
};

template <typename _Scalar>
struct DynamicsFiniteDifferenceHelper
    : internal::finite_difference_impl<_Scalar, DynamicsModelTpl> {
  using Scalar = _Scalar;

  using DynamicsModel = DynamicsModelTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Base = internal::finite_difference_impl<Scalar, DynamicsModelTpl>;
  using Base::computeJacobians;
  using Base::computeVectorHessianProducts;
  using Base::evaluate;

  PROXDDP_DYNAMIC_TYPEDEFS(_Scalar);

  DynamicsFiniteDifferenceHelper(shared_ptr<Manifold> space,
                                 shared_ptr<DynamicsModel> func,
                                 const Scalar fd_eps)
      : DynamicsModel(space, func->nu, space), Base(space, func, fd_eps) {}
};

template <typename Scalar>
struct CostFiniteDifferenceHelper : CostAbstractTpl<Scalar> {
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostBase = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  using CostBase::space;

  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);

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

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
extern template struct FiniteDifferenceHelper<context::Scalar>;
extern template struct DynamicsFiniteDifferenceHelper<context::Scalar>;
extern template struct CostFiniteDifferenceHelper<context::Scalar>;
#endif

} // namespace autodiff
} // namespace aligator
