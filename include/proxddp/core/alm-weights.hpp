#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {

/// @brief  Weighting strategy for the constraints in a stack.
template <typename Scalar> struct ConstraintProximalScalerTpl {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using DiagonalMatrix = Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>;

  ConstraintProximalScalerTpl(const ConstraintStack &constraints,
                              const Scalar &mu)
      : constraints_(&constraints), mu_(&mu), weights_(constraints.size()),
        dmatrix_(constraints.totalDim()) {
    weights_.setOnes();
    initMatrix();
  }

  std::size_t size() const { return constraints_->size(); }
  Scalar get(std::size_t j) const { return weights_[(long)j] * mu(); }

  Scalar inv(std::size_t j) const { return 1. / get(j); }
  void setWeight(const Scalar w, std::size_t j) {
    assert(j < weights_.size());
    weights_[(long)j] = w;
    auto &d = dmatrix_.diagonal();
    constraints_->segmentByConstraint(d, j).setConstant(get(j));
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

  /// Get the diagonal matrix corresponding to the ALM parameters.
  const DiagonalMatrix &matrix() const { return dmatrix_; }

private:
  void initMatrix() {
    const ConstraintStack &stack = *constraints_;
    auto &d = dmatrix_.diagonal();
    for (std::size_t j = 0; j < stack.size(); ++j) {
      stack.segmentByConstraint(d, j).setConstant(get(j));
    }
  }
  Scalar mu() const { return *mu_; }
  const ConstraintStack *constraints_;
  const Scalar *mu_;
  VectorXs weights_;
  DiagonalMatrix dmatrix_;
};

template <typename Scalar>
void ConstraintProximalScalerTpl<Scalar>::applyDefaultStrategy() {
  setWeight(1e-3, 0);
  for (std::size_t j = 1; j < constraints_->size(); j++) {
    setWeight(100., j);
  }
}

} // namespace proxddp
