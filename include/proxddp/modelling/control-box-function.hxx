#pragma once

#include "proxddp/modelling/control-box-function.hpp"

namespace proxddp {
template <typename Scalar>
ControlBoxFunctionTpl<Scalar>::ControlBoxFunctionTpl(const int ndx,
                                                     const VectorXs umin,
                                                     const VectorXs umax)
    : Base(ndx, umin.size(), ndx, 2 * umin.size()), umin_(umin), umax_(umax) {
  assert(umin.size() == umax.size() &&
         "Size of umin and umax should be the same!");
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
                                             const ConstVectorRef &,
                                             Data &data) const {
  data.value_.head(this->nu) = umin_ - u;
  data.value_.tail(this->nu) = u - umax_;
}

template <typename Scalar>
void ControlBoxFunctionTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     Data &data) const {
  data.Ju_.topRows(this->nu).diagonal().array() = static_cast<Scalar>(-1.);
  data.Ju_.bottomRows(this->nu).diagonal().array() = static_cast<Scalar>(1.);
}

template <typename Scalar>
shared_ptr<FunctionDataTpl<Scalar>>
ControlBoxFunctionTpl<Scalar>::createData() const {
  auto data =
      std::make_shared<Data>(this->ndx1, this->nu, this->ndx2, this->nr);
  data->Ju_.topRows(this->nu).diagonal().array() = static_cast<Scalar>(-1.);
  data->Ju_.bottomRows(this->nu).diagonal().array() = static_cast<Scalar>(1.);
  return data;
}
} // namespace proxddp
