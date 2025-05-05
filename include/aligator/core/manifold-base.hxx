/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/manifold-base.hpp"

namespace aligator {

/* Integrate */

template <typename Scalar>
void ManifoldAbstractTpl<Scalar>::integrate(const ConstVectorRef &x,
                                            const ConstVectorRef &v,
                                            VectorRef out) const {
  integrate_impl(x, v, out);
}

template <typename Scalar>
void ManifoldAbstractTpl<Scalar>::Jintegrate(const ConstVectorRef &x,
                                             const ConstVectorRef &v,
                                             MatrixRef Jout, int arg) const {
  Jintegrate_impl(x, v, Jout, arg);
}

/* Difference */

template <typename Scalar>
void ManifoldAbstractTpl<Scalar>::difference(const ConstVectorRef &x0,
                                             const ConstVectorRef &x1,
                                             VectorRef out) const {
  difference_impl(x0, x1, out);
}

template <typename Scalar>
void ManifoldAbstractTpl<Scalar>::Jdifference(const ConstVectorRef &x0,
                                              const ConstVectorRef &x1,
                                              MatrixRef Jout, int arg) const {
  Jdifference_impl(x0, x1, Jout, arg);
}

template <typename Scalar>
void ManifoldAbstractTpl<Scalar>::interpolate(const ConstVectorRef &x0,
                                              const ConstVectorRef &x1,
                                              const Scalar &u,
                                              VectorRef out) const {
  interpolate_impl(x0, x1, u, out);
}

} // namespace aligator
