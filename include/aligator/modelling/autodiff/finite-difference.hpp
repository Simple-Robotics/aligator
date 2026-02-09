/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/manifold-base.hpp"

namespace aligator {
namespace autodiff {

namespace internal {

template <typename _Scalar, template <typename> class _BaseTpl>
struct finite_diff_traits;

template <typename Scalar> struct finite_diff_traits<Scalar, StageFunctionTpl> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  struct Data : StageFunctionDataTpl<Scalar> {
    using SFD = StageFunctionDataTpl<Scalar>;
    using SFD::ndx1;
    using SFD::nu;
    shared_ptr<SFD> data_0;
    shared_ptr<SFD> data_1;
    VectorXs dx, du;
    VectorXs xp, up;

    template <typename U>
    Data(U const &model)
        : SFD(*model.func_)
        , data_0(model.func_->createData())
        , data_1(model.func_->createData())
        , dx(ndx1)
        , du(nu)
        , xp(model.nx1)
        , up(model.nu) {
      dx.setZero();
      du.setZero();
    }
  };

  struct Args {
    ConstVectorRef x;
    ConstVectorRef u;
  };
};

template <typename Scalar>
struct finite_diff_traits<Scalar, ExplicitDynamicsModelTpl> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  struct Data : ExplicitDynamicsDataTpl<Scalar> {
    using EDD = ExplicitDynamicsDataTpl<Scalar>;
    using EDD::ndx1;
    using EDD::ndx2;
    using EDD::nu;
    shared_ptr<EDD> data_0;
    shared_ptr<EDD> data_1;
    VectorXs dx, du;
    VectorXs xp, up;
    VectorXs dxnext;

    template <typename U>
    Data(U const &model)
        : EDD(*model.func_)
        , data_0(model.func_->createData())
        , data_1(model.func_->createData())
        , dx(ndx1)
        , du(nu)
        , xp(model.nx1)
        , up(model.nu)
        , dxnext(ndx2) {
      dx.setZero();
      du.setZero();
      dxnext.setZero();
    }
  };

  struct Args {
    ConstVectorRef x;
    ConstVectorRef u;
  };
};

/// @brief Implementation details for finite-differencing.
template <typename _Scalar, template <typename> class _BaseTpl>
struct finite_difference_impl : finite_diff_traits<_Scalar, _BaseTpl> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Traits = finite_diff_traits<Scalar, _BaseTpl>;
  using Data = typename Traits::Data;
  using Args = typename Traits::Args;
  using Base = _BaseTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  static_assert(std::is_base_of_v<BaseData, Data>);

  xyz::polymorphic<Manifold> space_;
  xyz::polymorphic<Base> func_;
  Scalar fd_eps;
  int nx1, nu, nx2;

  static constexpr bool IsStage =
      std::is_same_v<Base, StageFunctionTpl<Scalar>>;
  static constexpr bool IsExplicitDynamics =
      std::is_same_v<Base, ExplicitDynamicsModelTpl<Scalar>>;

  static_assert(IsStage || IsExplicitDynamics,
                "Unsupported finite_difference_impl base.");

  template <typename U = Base,
            std::enable_if_t<std::is_same_v<U, StageFunctionTpl<Scalar>>, int> =
                0>
  finite_difference_impl(xyz::polymorphic<Manifold> space,
                         xyz::polymorphic<U> func, const Scalar fd_eps)
      : space_(space)
      , func_(func)
      , fd_eps(fd_eps)
      , nx1(space->nx())
      , nx2(space->nx()) {}

  template <typename U = Base,
            std::enable_if_t<
                std::is_same_v<U, ExplicitDynamicsModelTpl<Scalar>>, int> = 0>
  finite_difference_impl(xyz::polymorphic<Manifold> space,
                         xyz::polymorphic<U> func, const Scalar fd_eps)
      : space_(space)
      , func_(func)
      , fd_eps(fd_eps)
      , nx1(space->nx())
      , nu(func->nu)
      , nx2(func->space_next().nx()) {}

  /// @details The @p y parameter is provided as a pointer, since is can be
  /// null.
  void evaluateImpl(const Args &args, BaseData &data) const {
    Data &d = static_cast<Data &>(data);
    assert(d.data_0);
    assert(d.data_1);
    if constexpr (IsExplicitDynamics) {
      func_->forward(args.x, args.u, *d.data_0);
      d.xnext_ = d.data_0->xnext_;
    } else {
      func_->evaluate(args.x, args.u, *d.data_0);
      d.value_ = d.data_0->value_;
    }
  }

  void computeJacobiansImpl(const Args &args, BaseData &data) const {
    Data &d = static_cast<Data &>(data);
    assert(d.data_0);
    assert(d.data_1);

    auto output_ref = [](auto &data_ref) -> VectorXs & {
      if constexpr (IsExplicitDynamics) {
        return data_ref.xnext_;
      } else {
        return data_ref.value_;
      }
    };

    VectorXs &v0 = output_ref(*d.data_0);
    VectorXs &vp = output_ref(*d.data_1);

    const int ndx1 = [&]() {
      if constexpr (IsExplicitDynamics) {
        return func_->ndx1();
      } else {
        return func_->ndx1;
      }
    }();

    for (int i = 0; i < ndx1; i++) {
      d.dx[i] = fd_eps;
      space_->integrate(args.x, d.dx, d.xp);
      if constexpr (IsExplicitDynamics) {
        func_->forward(d.xp, args.u, *d.data_1);
        func_->space_next().difference(v0, vp, d.dxnext);
        data.Jx().col(i) = d.dxnext / fd_eps;
      } else {
        func_->evaluate(d.xp, args.u, *d.data_1);
        data.Jx_.col(i) = (vp - v0) / fd_eps;
      }
      d.dx[i] = 0.;
    }

    for (int i = 0; i < func_->nu; i++) {
      d.du[i] = fd_eps;
      d.up = args.u + d.du;
      if constexpr (IsExplicitDynamics) {
        func_->forward(args.x, d.up, *d.data_1);
        func_->space_next().difference(v0, vp, d.dxnext);
        data.Ju().col(i) = d.dxnext / fd_eps;
      } else {
        func_->evaluate(args.x, d.up, *d.data_1);
        data.Ju_.col(i) = (vp - v0) / fd_eps;
      }
      d.du[i] = 0.;
    }
  }

  void computeVectorHessianProductsImpl(const ConstVectorRef &,
                                        const ConstVectorRef &,
                                        const ConstVectorRef &,
                                        const ConstVectorRef &,
                                        BaseData &) const {}

  shared_ptr<BaseData> createDataImpl() const {
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
struct FiniteDifferenceHelper : StageFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  using Impl = internal::finite_difference_impl<Scalar, StageFunctionTpl>;

  using StageFunction = StageFunctionTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Data = typename Impl::Data;
  using BaseData = StageFunctionDataTpl<Scalar>;

  ALIGATOR_DYNAMIC_TYPEDEFS(_Scalar);

  FiniteDifferenceHelper(xyz::polymorphic<Manifold> space,
                         xyz::polymorphic<StageFunction> func,
                         const Scalar fd_eps)
      : StageFunction(func->ndx1, func->nu, func->nr)
      , impl(space, func, fd_eps) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const {
    impl.evaluateImpl({x, u}, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const {
    impl.computeJacobiansImpl({x, u}, data);
  }

  shared_ptr<BaseData> createData() const { return impl.createDataImpl(); }

private:
  Impl impl;
};

template <typename _Scalar>
struct DynamicsFiniteDifferenceHelper : ExplicitDynamicsModelTpl<_Scalar> {
  using Scalar = _Scalar;

  using DynamicsModel = ExplicitDynamicsModelTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Impl =
      internal::finite_difference_impl<Scalar, ExplicitDynamicsModelTpl>;
  using Data = typename Impl::Data;
  using BaseData = ExplicitDynamicsDataTpl<Scalar>;

  ALIGATOR_DYNAMIC_TYPEDEFS(_Scalar);

  DynamicsFiniteDifferenceHelper(xyz::polymorphic<Manifold> space,
                                 xyz::polymorphic<DynamicsModel> func,
                                 const Scalar fd_eps)
      : DynamicsModel(space, func->nu)
      , impl(space, func, fd_eps) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const {
    impl.evaluateImpl({x, u}, data);
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const {
    impl.computeJacobiansImpl({x, u}, data);
  }

  shared_ptr<BaseData> createData() const { return impl.createDataImpl(); }

private:
  Impl impl;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct FiniteDifferenceHelper<context::Scalar>;
extern template struct DynamicsFiniteDifferenceHelper<context::Scalar>;
#endif

} // namespace autodiff
} // namespace aligator
