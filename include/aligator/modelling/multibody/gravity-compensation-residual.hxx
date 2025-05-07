#include "gravity-compensation-residual.hpp"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace aligator {

template <typename Scalar>
GravityCompensationResidualTpl<Scalar>::GravityCompensationResidualTpl(
    int ndx, const MatrixXs &actuation_matrix, const Model &model)
    : Base(ndx, (int)actuation_matrix.cols(), model.nv)
    , pin_model_(model)
    , actuation_matrix_(actuation_matrix)
    , use_actuation_matrix(true) {}

template <typename Scalar>
GravityCompensationResidualTpl<Scalar>::GravityCompensationResidualTpl(
    int ndx, const Model &model)
    : Base(ndx, model.nv, model.nv)
    , pin_model_(model)
    , actuation_matrix_()
    , use_actuation_matrix(false) {}

template <typename Scalar>
void GravityCompensationResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                      const ConstVectorRef &u,
                                                      BaseData &data_) const {
  Data &data = static_cast<Data &>(data_);
  const ConstVectorRef q = x.head(pin_model_.nq);
  data.value_ =
      -pinocchio::computeGeneralizedGravity(pin_model_, data.pin_data_, q);
  if (use_actuation_matrix) {
    data.value_ += actuation_matrix_ * u;
  } else {
    data.value_ += u;
  }
}

template <typename Scalar>
void GravityCompensationResidualTpl<Scalar>::computeJacobians(
    const ConstVectorRef &x, const ConstVectorRef & /*u*/,
    BaseData &data_) const {
  Data &data = static_cast<Data &>(data_);
  ConstVectorRef q = x.head(pin_model_.nq);
  pinocchio::computeGeneralizedGravityDerivatives(
      pin_model_, data.pin_data_, q,
      pinocchio::make_ref(data.gravity_partial_dq_));
  if (use_actuation_matrix) {
    data_.Ju_ = actuation_matrix_;
  } else {
    data_.Ju_.setIdentity();
  }
  data.Jx_.leftCols(pin_model_.nv) = -data.gravity_partial_dq_;
}

template <typename Scalar>
GravityCompensationResidualTpl<Scalar>::Data::Data(
    const GravityCompensationResidualTpl &resdl)
    : BaseData(resdl)
    , pin_data_(resdl.pin_model_)
    , tmp_torque_(resdl.pin_model_.nv)
    , gravity_partial_dq_(resdl.pin_model_.nv, resdl.pin_model_.nv) {}

template <typename Scalar>
auto GravityCompensationResidualTpl<Scalar>::createData() const
    -> shared_ptr<BaseData> {
  return std::make_shared<Data>(*this);
}
} // namespace aligator
