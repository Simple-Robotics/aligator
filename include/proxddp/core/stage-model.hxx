#include "proxddp/core/stage-model.hpp"

namespace proxddp {
template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(const ManifoldPtr &space1, const int nu,
                                     const ManifoldPtr &space2,
                                     const CostPtr &cost,
                                     const shared_ptr<Dynamics> &dyn_model)
    : xspace_(space1), xspace_next_(space2),
      uspace_(std::make_shared<proxnlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>>(
          nu)),
      cost_(cost) {
  using EqualitySet = proxnlp::EqualityConstraint<Scalar>;
  constraints_manager.push_back(
      Constraint{dyn_model, std::make_shared<EqualitySet>()});
}

template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(const ManifoldPtr &space, const int nu,
                                     const CostPtr &cost,
                                     const shared_ptr<Dynamics> &dyn_model)
    : StageModelTpl(space, nu, space, cost, dyn_model) {}

template <typename Scalar> inline int StageModelTpl<Scalar>::numPrimal() const {
  return this->nu() + this->ndx2();
}

template <typename Scalar> inline int StageModelTpl<Scalar>::numDual() const {
  return constraints_manager.totalDim();
}

template <typename Scalar>
void StageModelTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                     const ConstVectorRef &u,
                                     const ConstVectorRef &y,
                                     Data &data) const {
  for (std::size_t j = 0; j < numConstraints(); j++) {
    // calc on constraint
    const Constraint &cstr = constraints_manager[j];
    cstr.func_->evaluate(x, u, y, *data.constraint_data[j]);
  }
  cost_->evaluate(x, u, *data.cost_data);
}

template <typename Scalar>
void StageModelTpl<Scalar>::computeDerivatives(const ConstVectorRef &x,
                                               const ConstVectorRef &u,
                                               const ConstVectorRef &y,
                                               Data &data) const {
  for (std::size_t j = 0; j < numConstraints(); j++) {
    // calc on constraint
    const Constraint &cstr = constraints_manager[j];
    cstr.func_->computeJacobians(x, u, y, *data.constraint_data[j]);
  }
  cost_->computeGradients(x, u, *data.cost_data);
  cost_->computeHessians(x, u, *data.cost_data);
}

template <typename Scalar>
StageDataTpl<Scalar>::StageDataTpl(const StageModel &stage_model)
    : constraint_data(stage_model.numConstraints()),
      cost_data(std::move(stage_model.cost_->createData())) {
  const std::size_t nc = stage_model.numConstraints();
  for (std::size_t j = 0; j < nc; j++) {
    const shared_ptr<StageFunctionTpl<Scalar>> &func =
        stage_model.constraints_manager[j].func_;
    constraint_data[j] = std::move(func->createData());
  }
}
} // namespace proxddp
