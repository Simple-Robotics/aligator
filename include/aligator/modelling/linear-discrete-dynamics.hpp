#pragma once

#include "aligator/core/explicit-dynamics.hpp"
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

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
  using DynData = DynamicsDataTpl<Scalar>;
  using Data = ExplicitDynamicsDataTpl<Scalar>;
  using VectorSpaceType =
      proxsuite::nlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  /// @brief Constructor with state manifold and matrices.
  LinearDiscreteDynamicsTpl(const MatrixXs &A, const MatrixXs &B,
                            const VectorXs &c)
      : Base(VectorSpaceType((int)A.cols()), (int)B.cols()), A_(A), B_(B),
        c_(c) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               Data &data) const {
    data.xnext_ = A_ * x + B_ * u + c_;
  }

  void dForward(const ConstVectorRef &, const ConstVectorRef &, Data &) const {}

  shared_ptr<DynData> createData() const {
    auto data =
        std::make_shared<Data>(this->ndx1, this->nu, this->nx2(), this->ndx2);
    data->Jx_ = A_;
    data->Ju_ = B_;
    return data;
  }
};

} // namespace dynamics

} // namespace aligator
