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
      : constraints_(&constraints), mu_(&mu), weights_(constraints.size()),
        scaleMatDiag_(constraints.getDims()) {
    weights_.setOnes();
    initMatrix();
  }

  std::size_t size() const { return constraints_->size(); }
  Scalar get(std::size_t j) const { return weights_[(long)j] * mu(); }

  void setWeight(const Scalar w, std::size_t j) {
    assert(j < (std::size_t)weights_.size());
    weights_[(long)j] = w;
    scaleMatDiag_[j].array() = get(j);
  }

  VectorXs &getWeights() { return weights_; }
  const VectorXs &getWeights() const { return weights_; }
  /// Set all weights at once
  template <typename D> void setWeights(const Eigen::MatrixBase<D> &w) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(D)
    assert(w.size() == weights_.size());
    weights_ = w;
    initMatrix(); // reinitialize matrix
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
    return m.dot(diagMatrix().asDiagonal() * m);
  }

  const VectorXs &diagMatrix() const { return scaleMatDiag_.matrix(); }

private:
  using BlkVec = BlkMatrix<VectorXs, -1, 1>;
  /// Initialize the penalty weight matrix
  void initMatrix();
  Scalar mu() const { return *mu_; }
  const ConstraintStack *constraints_;
  const Scalar *mu_;
  VectorXs weights_;
  /// Diagonal for the scaling matrix
  BlkVec scaleMatDiag_;
};

template <typename Scalar>
void ConstraintProximalScalerTpl<Scalar>::initMatrix() {
  for (std::size_t j = 0; j < constraints_->size(); ++j)
    scaleMatDiag_[j].array() = get(j);
}

} // namespace aligator
