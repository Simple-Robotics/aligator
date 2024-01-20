#pragma once

#include "aligator/fwd.hpp"
#include "aligator/gar/blk-matrix.hpp"

namespace aligator {

/// @brief  Weighting strategy for the constraints in a stack.
template <typename Scalar> class ConstraintProximalScalerTpl {
public:
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintStack = ConstraintStackTpl<Scalar>;

  ConstraintProximalScalerTpl(const ConstraintStack &constraints,
                              const Scalar &mu)
      : constraints_(&constraints), mu_(&mu),
        scalingMatrix_(constraints.dims()) {
    scalingMatrix_.matrix().setOnes();
  }

  std::size_t size() const { return constraints_->size(); }

  void setWeight(const Scalar w, std::size_t j) {
    scalingMatrix_[j].array() = w;
  }

  /// Set all weights at once
  template <typename D> void setWeights(const Eigen::MatrixBase<D> &weights) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(D)
    for (size_t i = 0; i < constraints_->size(); i++) {
      scalingMatrix_[i].array() = weights[long(i)];
    }
  }

  /// Apply weighted penalty matrix
  template <typename D> auto apply(const Eigen::MatrixBase<D> &m) const {
    return diagMatrix().asDiagonal() * m;
  }

  template <typename D> auto applyInverse(const Eigen::MatrixBase<D> &m) const {
    return diagMatrix().asDiagonal().inverse() * m;
  }

  template <typename D>
  Scalar weightedNorm(const Eigen::MatrixBase<D> &m) const {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(D)
    return m.dot(apply(m));
  }

  inline auto diagMatrix() const { return mu() * scalingMatrix_.matrix(); }

private:
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;
  Scalar mu() const { return *mu_; }
  const ConstraintStack *constraints_;
  /// Solver mu parameter
  const Scalar *mu_;
  /// Diagonal for the scaling matrix
  BlkVec scalingMatrix_;
};

} // namespace aligator
