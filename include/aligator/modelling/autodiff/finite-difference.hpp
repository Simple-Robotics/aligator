/// @file   Helper structs to define derivatives on a function
///         through finite differences.
#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/dynamics.hpp"
#include <proxsuite-nlp/manifold-base.hpp>
#include <boost/mpl/bool.hpp>

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
        : SFD(model.func_), data_0(model.func_->createData()),
          data_1(model.func_->createData()), dx(ndx1), du(nu), xp(model.nx1),
          up(model.nu) {
      dx.setZero();
      du.setZero();
    }
  };

  struct Args {
    ConstVectorRef x;
    ConstVectorRef u;
  };
};

template <typename Scalar> struct finite_diff_traits<Scalar, DynamicsModelTpl> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  struct Data : DynamicsDataTpl<Scalar> {
    using DD = DynamicsDataTpl<Scalar>;
    using DD::ndx1;
    using DD::ndx2;
    using DD::nu;
    shared_ptr<DD> data_0;
    shared_ptr<DD> data_1;
    VectorXs dx, du, dy;
    VectorXs xp, up, yp;

    template <typename U>
    Data(U const &model)
        : DD(model.func_), data_0(model.func_->createData()),
          data_1(model.func_->createData()), dx(ndx1), du(nu), dy(ndx2),
          xp(model.nx1), up(model.nu), yp(model.nx2) {
      dx.setZero();
      du.setZero();
      dy.setZero();
    }
  };

  struct Args {
    ConstVectorRef x;
    ConstVectorRef u;
    ConstVectorRef y;
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

  xyz::polymorphic<Manifold> space_;
  xyz::polymorphic<Base> func_;
  Scalar fd_eps;
  int nx1, nu, nx2;

  static constexpr bool IsDynamics =
      std::is_same_v<Base, DynamicsModelTpl<Scalar>>;

  template <typename U = Base, class = std::enable_if_t<
                                   std::is_same_v<U, StageFunctionTpl<Scalar>>>>
  finite_difference_impl(xyz::polymorphic<Manifold> space,
                         xyz::polymorphic<U> func, const Scalar fd_eps)
      : space_(space), func_(func), fd_eps(fd_eps), nx1(space->nx()),
        nx2(space->nx()) {}

  template <typename U = Base, class = std::enable_if_t<
                                   std::is_same_v<U, DynamicsModelTpl<Scalar>>>>
  finite_difference_impl(xyz::polymorphic<Manifold> space,
                         xyz::polymorphic<U> func, const Scalar fd_eps,
                         boost::mpl::false_ = {})
      : space_(space), func_(func), fd_eps(fd_eps), nx1(space->nx()),
        nu(func->nu), nx2(space->nx()) {}

  /// @details The @p y parameter is provided as a pointer, since is can be
  /// null.
  void evaluateImpl(const Args &args, BaseData &data) const {
    Data &d = static_cast<Data &>(data);
    if constexpr (IsDynamics) {
      func_->evaluate(args.x, args.u, args.y, *d.data_0);
    } else {
      func_->evaluate(args.x, args.u, *d.data_0);
    }
    d.value_ = d.data_0->value_;
  }

  void computeJacobiansImpl(const Args &args, BaseData &data) const {
    Data &d = static_cast<Data &>(data);

    VectorXs &v0 = d.data_0->value_;
    VectorXs &vp = d.data_1->value_;

    for (int i = 0; i < func_->ndx1; i++) {
      d.dx[i] = fd_eps;
      space_->integrate(args.x, d.dx, d.xp);
      if constexpr (IsDynamics) {
        func_->evaluate(d.xp, args.u, args.y, *d.data_1);
      } else {
        func_->evaluate(d.xp, args.u, *d.data_1);
      }
      data.Jx_.col(i) = (vp - v0) / fd_eps;
      d.dx[i] = 0.;
    }

    if constexpr (IsDynamics) {
      for (int i = 0; i < func_->ndx2; i++) {
        d.dy[i] = fd_eps;
        space_->integrate(args.y, d.dy, d.yp);
        func_->evaluate(args.x, args.u, d.yp, *d.data_1);
        data.Jy_.col(i) = (vp - v0) / fd_eps;
        d.dy[i] = 0.;
      }
    }

    for (int i = 0; i < func_->nu; i++) {
      d.du[i] = fd_eps;
      d.up = args.u + d.du;
      if constexpr (IsDynamics) {
        func_->evaluate(args.x, d.up, args.y, *d.data_1);
      } else {
        func_->evaluate(args.x, d.up, *d.data_1);
      }
      data.Ju_.col(i) = (vp - v0) / fd_eps;
      d.du[i] = 0.;
    }
  }

  void computeVectorHessianProductsImpl(const ConstVectorRef &,
                                        const ConstVectorRef &,
                                        const ConstVectorRef &,
                                        const ConstVectorRef &,
                                        BaseData &) const {}

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
      : StageFunction(func->ndx1, func->nu, func->nr),
        impl(space, func, fd_eps) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const {
    impl.evaluateImpl({x, u}, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const {
    impl.computeJacobiansImpl({x, u}, data);
  }

  Impl impl;
};

template <typename _Scalar>
struct DynamicsFiniteDifferenceHelper : DynamicsModelTpl<_Scalar> {
  using Scalar = _Scalar;

  using DynamicsModel = DynamicsModelTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Impl = internal::finite_difference_impl<Scalar, DynamicsModelTpl>;
  using Data = typename Impl::Data;
  using BaseData = DynamicsDataTpl<Scalar>;

  ALIGATOR_DYNAMIC_TYPEDEFS(_Scalar);

  DynamicsFiniteDifferenceHelper(xyz::polymorphic<Manifold> space,
                                 xyz::polymorphic<DynamicsModel> func,
                                 const Scalar fd_eps)
      : DynamicsModel(space, func->nu, space), impl(space, func, fd_eps) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &xn, BaseData &data) const {
    impl.evaluateImpl({x, u, xn}, data);
  }

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &xn, BaseData &data) const {
    impl.computeJacobiansImpl({x, u, xn}, data);
  }

  Impl impl;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct FiniteDifferenceHelper<context::Scalar>;
extern template struct DynamicsFiniteDifferenceHelper<context::Scalar>;
#endif

} // namespace autodiff
} // namespace aligator
