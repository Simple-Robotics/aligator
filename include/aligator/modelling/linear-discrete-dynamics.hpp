#pragma once

#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/vector-space.hpp"

namespace aligator {

namespace dynamics {

/// @brief Discrete explicit linear dynamics.
template <typename _Scalar>
struct LinearDiscreteDynamicsTpl : ExplicitDynamicsModelTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  const MatrixXs A_;
  const MatrixXs B_;
  const VectorXs c_;

  using Base = ExplicitDynamicsModelTpl<Scalar>;
  using Data = ExplicitDynamicsDataTpl<Scalar>;
  using VectorSpaceType = VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  /// @brief Constructor with state manifold and matrices.
  LinearDiscreteDynamicsTpl(const MatrixXs &A, const MatrixXs &B,
                            const VectorXs &c)
      : Base(VectorSpaceType((int)A.cols()), (int)B.cols())
      , A_(A)
      , B_(B)
      , c_(c) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               Data &data) const {
    data.xnext_ = c_;
    data.xnext_.noalias() += A_ * x;
    data.xnext_.noalias() += B_ * u;
  }

  void dForward(const ConstVectorRef &, const ConstVectorRef &, Data &) const {}

  shared_ptr<Data> createData() const {
    shared_ptr<Data> data = Base::createData();
    data->Jx() = A_;
    data->Ju() = B_;
    return data;
  }
};

} // namespace dynamics

} // namespace aligator
