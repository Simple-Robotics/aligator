/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
#pragma once

#include "aligator/core/constraint-set.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {
template <typename Derived>
auto blockMatrixGetRow(const Eigen::MatrixBase<Derived> &matrix,
                       const std::vector<Eigen::Index> &rowBlockSizes,
                       std::size_t rowIdx) {
  Eigen::Index startIdx = 0;
  for (std::size_t kk = 0; kk < rowIdx; kk++) {
    startIdx += rowBlockSizes[kk];
  }
  return matrix.const_cast_derived().middleRows(startIdx,
                                                rowBlockSizes[rowIdx]);
}

template <typename Derived>
auto blockVectorGetRow(const Eigen::MatrixBase<Derived> &matrix,
                       const std::vector<Eigen::Index> &blockSizes,
                       std::size_t blockIdx) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  Eigen::Index startIdx = 0;
  for (std::size_t kk = 0; kk < blockIdx; kk++) {
    startIdx += blockSizes[kk];
  }
  return matrix.const_cast_derived().segment(startIdx, blockSizes[blockIdx]);
}

/// @brief Cartesian product of multiple constraint sets.
/// This class makes computing multipliers and Jacobian matrix projections more
/// convenient.
/// @warning This struct contains a non-owning vector of its component sets.
template <typename Scalar>
struct ConstraintSetProductTpl : ConstraintSetTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ConstraintSetTpl<Scalar>;
  using ActiveType = typename Base::ActiveType;

  ConstraintSetProductTpl(const std::vector<xyz::polymorphic<Base>> components,
                          const std::vector<Eigen::Index> &blockSizes)
      : m_components(components), m_blockSizes(blockSizes) {
    if (components.size() != blockSizes.size()) {
      ALIGATOR_RUNTIME_ERROR("Number of components and corresponding "
                             "block sizes should be the same.");
    }
  }

  ConstraintSetProductTpl(const ConstraintSetProductTpl &) = default;
  ConstraintSetProductTpl &operator=(const ConstraintSetProductTpl &) = default;
  ConstraintSetProductTpl(ConstraintSetProductTpl &&) = default;
  ConstraintSetProductTpl &operator=(ConstraintSetProductTpl &&) = default;

  Scalar evaluate(const ConstVectorRef &zproj) const override {
    Scalar res = 0.;
    for (std::size_t i = 0; i < m_components.size(); i++) {
      auto zblock = blockVectorGetRow(zproj, m_blockSizes, i);
      res += m_components[i]->evaluate(zblock);
    }
    return res;
  }

  void projection(const ConstVectorRef &z, VectorRef zout) const override {
    for (std::size_t i = 0; i < m_components.size(); i++) {
      ConstVectorRef inblock = blockVectorGetRow(z, m_blockSizes, i);
      VectorRef outblock = blockVectorGetRow(zout, m_blockSizes, i);
      m_components[i]->projection(inblock, outblock);
    }
  }

  void normalConeProjection(const ConstVectorRef &z,
                            VectorRef zout) const override {
    for (std::size_t i = 0; i < m_components.size(); i++) {
      ConstVectorRef inblock = blockVectorGetRow(z, m_blockSizes, i);
      VectorRef outblock = blockVectorGetRow(zout, m_blockSizes, i);
      m_components[i]->normalConeProjection(inblock, outblock);
    }
  }

  void applyProjectionJacobian(const ConstVectorRef &z,
                               MatrixRef Jout) const override {
    for (std::size_t i = 0; i < m_components.size(); i++) {
      ConstVectorRef inblock = blockVectorGetRow(z, m_blockSizes, i);
      MatrixRef outblock = blockMatrixGetRow(Jout, m_blockSizes, i);
      m_components[i]->applyProjectionJacobian(inblock, outblock);
    }
  }

  void applyNormalConeProjectionJacobian(const ConstVectorRef &z,
                                         MatrixRef Jout) const override {
    for (std::size_t i = 0; i < m_components.size(); i++) {
      ConstVectorRef inblock = blockVectorGetRow(z, m_blockSizes, i);
      MatrixRef outblock = blockMatrixGetRow(Jout, m_blockSizes, i);
      m_components[i]->applyNormalConeProjectionJacobian(inblock, outblock);
    }
  }

  void computeActiveSet(const ConstVectorRef &z,
                        Eigen::Ref<ActiveType> out) const override {
    for (std::size_t i = 0; i < m_components.size(); i++) {
      ConstVectorRef inblock = blockVectorGetRow(z, m_blockSizes, i);
      decltype(out) outblock = blockVectorGetRow(out, m_blockSizes, i);
      m_components[i]->computeActiveSet(inblock, outblock);
    }
  }

  const std::vector<xyz::polymorphic<Base>> &components() const {
    return m_components;
  }
  const std::vector<Eigen::Index> &blockSizes() const { return m_blockSizes; }

private:
  std::vector<xyz::polymorphic<Base>> m_components;
  std::vector<Eigen::Index> m_blockSizes;
};

} // namespace aligator
