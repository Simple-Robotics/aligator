#pragma once

#include "aligator/modelling/control-box-function.hpp"

namespace aligator {
template <typename Scalar>
ControlBoxFunctionTpl<Scalar>::ControlBoxFunctionTpl(const int ndx,
                                                     const VectorXs &umin,
                                                     const VectorXs &umax)
    : Base(ndx, (int)umin.size(), 2 * (int)umin.size()), umin_(umin),
      umax_(umax) {
  if (umin.size() != umax.size()) {
    ALIGATOR_DOMAIN_ERROR("Size of umin and umax should be the same!");
  }
}

template <typename Scalar>
ControlBoxFunctionTpl<Scalar>::ControlBoxFunctionTpl(const int ndx,
                                                     const int nu, Scalar umin,
                                                     Scalar umax)
    : ControlBoxFunctionTpl(ndx, VectorXs::Constant(nu, umin),
                            VectorXs::Constant(nu, umax)) {}

template <typename Scalar>
void ControlBoxFunctionTpl<Scalar>::evaluate(const ConstVectorRef &,
                                             const ConstVectorRef &u,
                                             Data &data) const {
  data.value_.head(this->nu) = umin_ - u;
  data.value_.tail(this->nu) = u - umax_;
}

template <typename Scalar>
void ControlBoxFunctionTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     Data &data) const {
  data.Ju_.topRows(this->nu).diagonal().array() = static_cast<Scalar>(-1.);
  data.Ju_.bottomRows(this->nu).diagonal().array() = static_cast<Scalar>(1.);
}

template <typename Scalar>
shared_ptr<StageFunctionDataTpl<Scalar>>
ControlBoxFunctionTpl<Scalar>::createData() const {
  auto data = std::make_shared<Data>(*this);
  data->Ju_.topRows(this->nu).diagonal().array() = static_cast<Scalar>(-1.);
  data->Ju_.bottomRows(this->nu).diagonal().array() = static_cast<Scalar>(1.);
  return data;
}
} // namespace aligator
