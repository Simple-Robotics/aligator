#pragma once

#include "aligator/core/manifold-base.hpp"

namespace aligator {

/// @brief Tangent bundle of a base manifold M.
template <class Base>
struct TangentBundleTpl : ManifoldAbstractTpl<typename Base::Scalar> {
protected:
  Base base_;

public:
  using Self = TangentBundleTpl<Base>;
  using Scalar = typename Base::Scalar;
  static constexpr int Options = Base::Options;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ManifoldBase = ManifoldAbstractTpl<Scalar>;
  using ManifoldBase::ndx;
  using ManifoldBase::nx;

  /// Constructor using base space instance.
  TangentBundleTpl(const Base &base)
      : ManifoldBase(base.nx() + base.ndx(), 2 * base.ndx())
      , base_(base) {}

  /// Constructor using base space constructor.
  template <typename... BaseCtorArgs>
  TangentBundleTpl(BaseCtorArgs... args)
      : ManifoldBase(0, 0)
      , base_(args...) {
    this->nx_ = base_.nx() + base_.ndx();
    this->ndx_ = 2 * base_.ndx();
  }

  bool isNormalized(const ConstVectorRef &x) const {
    auto p = getBasePoint(x);
    return base_.isNormalized(p);
  }

  const Base &getBaseSpace() const { return base_; }

  /// Get base point of an element of the tangent bundle.
  /// This map is exactly the natural projection.
  template <typename Point>
  typename Point::ConstSegmentReturnType
  getBasePoint(const Eigen::MatrixBase<Point> &x) const {
    return x.head(base_.nx());
  }

  template <typename Point>
  typename Point::SegmentReturnType
  getBasePointWrite(const Eigen::MatrixBase<Point> &x) const {
    return x.const_cast_derived().head(base_.nx());
  }

  template <typename Tangent>
  typename Tangent::ConstSegmentReturnType
  getBaseTangent(const Tangent &v) const {
    return v.head(base_.ndx());
  }

  template <typename Tangent>
  typename Tangent::SegmentReturnType
  getTangentHeadWrite(const Eigen::MatrixBase<Tangent> &v) const {
    return v.const_cast_derived().head(base_.ndx());
  }

  template <typename Jac>
  Eigen::Block<Jac, Eigen::Dynamic, Eigen::Dynamic>
  getBaseJacobian(const Eigen::MatrixBase<Jac> &J) const {
    return J.const_cast_derived().topLeftCorner(base_.ndx(), base_.ndx());
  }

protected:
  /// @name   Implementations of operators

  void neutral_impl(VectorRef out) const;
  void rand_impl(VectorRef out) const;

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &dx,
                      VectorRef out) const;

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const;

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const;

  void JintegrateTransport_impl(const ConstVectorRef &x,
                                const ConstVectorRef &v, MatrixRef Jout,
                                int arg) const;

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const;

  void interpolate_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        const Scalar &u, VectorRef out) const;
};

} // namespace aligator

#include "aligator/modelling/spaces/tangent-bundle.hxx"
