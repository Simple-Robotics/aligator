#pragma once

#include "aligator/fwd.hpp"
#include "aligator/gar/blk-matrix.hpp"

#include <proxsuite-nlp/constraint-base.hpp>

namespace aligator {

/// @brief Cartesian product of multiple constraint sets.
/// This class makes computing multipliers and Jacobian matrix projections more
/// convenient.
/// @warning This struct containts a non-owning vector of its component sets.
template <typename Scalar>
struct ConstraintSetProductTpl : ConstraintSetBase<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ConstraintSetBase<Scalar>;
  using BlkVecViewMut = BlkMatrix<VectorRef, -1, 1>;
  using BlkVecView = BlkMatrix<ConstVectorRef, -1, 1>;
  using BlkMatViewMut = BlkMatrix<MatrixRef, -1, 1>;
  using ActiveType = typename Base::ActiveType;

  Scalar evaluate(const ConstVectorRef &zproj) const override {
    BlkVecView zprojView(zproj, nrs);
    Scalar res = 0.;
    for (std::size_t i = 0; i < components.size(); i++) {
      res += components[i]->evaluate(zprojView.blockSegment(i));
    }
    return res;
  }

  void projection(const ConstVectorRef &z, VectorRef zout) const override {
    BlkVecView zv(z, nrs);
    BlkVecViewMut zov(zout, nrs);
    for (std::size_t i = 0; i < components.size(); i++) {
      components[i]->projection(zv[i], zov[i]);
    }
  }

  void normalConeProjection(const ConstVectorRef &z,
                            VectorRef zout) const override {
    BlkVecView zv(z, nrs);
    BlkVecViewMut zov(zout, nrs);
    for (std::size_t i = 0; i < components.size(); i++) {
      components[i]->normalConeProjection(zv[i], zov[i]);
    }
  }

  void applyProjectionJacobian(const ConstVectorRef &z,
                               MatrixRef Jout) const override {
    BlkVecView zv(z, nrs);
    BlkMatViewMut Jv(Jout, nrs, {1});
    for (std::size_t i = 0; i < components.size(); i++) {
      components[i]->applyProjectionJacobian(zv[i], Jv.blockRow(i));
    }
  }

  void computeActiveSet(const ConstVectorRef &z_,
                        Eigen::Ref<ActiveType> out_) const override {
    BlkVecView zv(z_, nrs);
    BlkMatrix<decltype(out_), -1, 1> outv(out_, nrs);
    for (std::size_t i = 0; i < components.size(); i++) {
      components[i]->computeActiveSet(zv[i], outv[i]);
    }
  }

  std::vector<Base *> components;
  std::vector<long> nrs;
};

} // namespace aligator
