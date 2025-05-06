#pragma once

#include "tangent-bundle.hpp"

namespace aligator {

template <class Base>
void TangentBundleTpl<Base>::neutral_impl(VectorRef out) const {
  out.setZero();
  out.head(base_.nx()) = base_.neutral();
}

template <class Base>
void TangentBundleTpl<Base>::rand_impl(VectorRef out) const {
  out.head(base_.nx()) = base_.rand();
  out.tail(base_.ndx()) = VectorXs::Random(base_.ndx());
}

/// Operators
template <class Base>
void TangentBundleTpl<Base>::integrate_impl(const ConstVectorRef &x,
                                            const ConstVectorRef &dx,
                                            VectorRef out) const {
  const int nv_ = base_.ndx();
  base_.integrate(getBasePoint(x), getBaseTangent(dx), out.head(base_.nx()));
  out.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

template <class Base>
void TangentBundleTpl<Base>::difference_impl(const ConstVectorRef &x0,
                                             const ConstVectorRef &x1,
                                             VectorRef out) const {
  const int nv_ = base_.ndx();
  out.resize(ndx());
  base_.difference(getBasePoint(x0), getBasePoint(x1), out.head(nv_));
  out.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
}

template <class Base>
void TangentBundleTpl<Base>::Jintegrate_impl(const ConstVectorRef &x,
                                             const ConstVectorRef &dx,
                                             MatrixRef J_, int arg) const {
  const int nv_ = base_.ndx();
  J_.resize(ndx(), ndx());
  J_.setZero();
  base_.Jintegrate(getBasePoint(x), getBaseTangent(dx), getBaseJacobian(J_),
                   arg);
  J_.bottomRightCorner(nv_, nv_).setIdentity();
}

template <class Base>
void TangentBundleTpl<Base>::Jdifference_impl(const ConstVectorRef &x0,
                                              const ConstVectorRef &x1,
                                              MatrixRef J_, int arg) const {
  const int nv_ = base_.ndx();
  J_.resize(ndx(), ndx());
  J_.setZero();
  base_.Jdifference(getBasePoint(x0), getBasePoint(x1), getBaseJacobian(J_),
                    arg);
  if (arg == 0) {
    J_.bottomRightCorner(nv_, nv_).diagonal().array() = Scalar(-1);
  } else if (arg == 1) {
    J_.bottomRightCorner(nv_, nv_).setIdentity();
  }
}

template <class Base>
void TangentBundleTpl<Base>::JintegrateTransport_impl(const ConstVectorRef &x,
                                                      const ConstVectorRef &v,
                                                      MatrixRef Jout,
                                                      int arg) const {
  const int nv_ = base_.ndx();
  base_.JintegrateTransport(getBasePoint(x), getBaseTangent(v),
                            Jout.topRows(nv_), arg);
}

template <class Base>
void TangentBundleTpl<Base>::interpolate_impl(const ConstVectorRef &x0,
                                              const ConstVectorRef &x1,
                                              const Scalar &u,
                                              VectorRef out) const {
  base_.interpolate(getBasePoint(x0), getBasePoint(x1), u,
                    getBasePointWrite(out));
  out.tail(base_.ndx()) =
      (Scalar(1.) - u) * getBaseTangent(x0) + u * getBaseTangent(x1);
}

} // namespace aligator
