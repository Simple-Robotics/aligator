#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {

/// @brief  Weighting strategy for the constraints in a stack.
template <typename Scalar> struct ConstraintProximalScalerTpl {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintStack = ConstraintStackTpl<Scalar>;

  ConstraintProximalScalerTpl(const ConstraintStack &constraints,
                              const Scalar &mu)
      : constraints_(&constraints), mu_(&mu), weights_(constraints.size()),
        scaleMatDiag_(constraints.totalDim()) {
    weights_.setOnes();
    initMatrix();
  }

  std::size_t size() const { return constraints_->size(); }
  Scalar get(std::size_t j) const { return weights_[(long)j] * mu(); }

  Scalar inv(std::size_t j) const { return 1. / get(j); }
  void setWeight(const Scalar w, std::size_t j) {
    assert(j < (std::size_t)weights_.size());
    weights_[(long)j] = w;
    constraints_->segmentByConstraint(scaleMatDiag_, j).setConstant(get(j));
  }

  /// Set all weights at once
  void setWeights(const ConstVectorRef &w) {
    assert(w.size() == weights_.size());
    weights_ = w;
    initMatrix(); // reinitialize matrix
  }

  /// For problem stages. Scale down non-dynamical constraints by 100.
  void applyDefaultStrategy();

  const VectorXs &getWeights() const { return weights_; }

  template <typename MatrixType>
  auto apply(const Eigen::MatrixBase<MatrixType> &m) const {
    return scaleMatDiag_.asDiagonal() * m;
  }

  auto matrix() { return scaleMatDiag_.asDiagonal(); }
  auto matrix() const { return scaleMatDiag_.asDiagonal(); }

private:
  void initMatrix() {
    for (std::size_t j = 0; j < constraints_->size(); ++j) {
      constraints_->segmentByConstraint(scaleMatDiag_, j).setConstant(get(j));
    }
  }
  Scalar mu() const { return *mu_; }
  const ConstraintStack *constraints_;
  const Scalar *mu_;
  VectorXs weights_;
  /// Diagonal for the scaling matrix
  VectorXs scaleMatDiag_;
};

template <typename Scalar>
void ConstraintProximalScalerTpl<Scalar>::applyDefaultStrategy() {
  setWeight(1e-3, 0);
  for (std::size_t j = 1; j < constraints_->size(); j++) {
    setWeight(100., j);
  }
}

} // namespace proxddp
