#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
  template<typename Scalar>
  StageModelTpl<Scalar>::
  StageModelTpl(const Manifold& space1,
                const int nu,
                const Manifold& space2,
                const CostBase& cost,
                const Dynamics& dyn_model)
    : xspace1_(space1)
    , xspace2_(space2)
    , uspace_(nu)
    , cost_(cost)
  {
    ConstraintPtr dynptr = std::make_shared<Constraint>(
      dyn_model, std::make_shared<proxnlp::EqualityConstraint<Scalar>>());
    constraints_manager.push_back(std::move(dynptr));
  }
  
  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numPrimal() const {
    return this->nu() + this->ndx2();
  }

  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numDual() const {
    return constraints_manager.totalDim();
  }

  template<typename Scalar>
  void StageModelTpl<Scalar>::evaluate(
    const ConstVectorRef& x,
    const ConstVectorRef& u,
    const ConstVectorRef& y,
    Data& data) const
  {
    cost_.evaluate(x, u, *data.cost_data);

    for (std::size_t i = 0; i < numConstraints(); i++)
    {
      // calc on constraint
      const auto& cstr = constraints_manager[i];
      cstr->func_.evaluate(x, u, y, *data.constraint_data[i]);
    }
  }

  template<typename Scalar>
  void StageModelTpl<Scalar>::computeDerivatives(
    const ConstVectorRef& x,
    const ConstVectorRef& u,
    const ConstVectorRef& y,
    Data& data) const
  {
    cost_.computeGradients(x, u, *data.cost_data);
    cost_.computeHessians (x, u, *data.cost_data);

    for (std::size_t i = 0; i < numConstraints(); i++)
    {
      // calc on constraint
      const ConstraintPtr& cstr = constraints_manager[i];
      cstr->func_.computeJacobians(x, u, y, *data.constraint_data[i]);
    }
  }

  template<typename Scalar>
  std::ostream& operator<<(std::ostream& oss, const StageModelTpl<Scalar>& stage)
  {
    oss << "StageModel { ";
    if (stage.ndx1() == stage.ndx2())
    {
      oss << "ndx: " << stage.ndx1() << ", "
          << "nu:  " << stage.nu();
    } else {
      oss << "ndx1:" << stage.ndx1() << ", "
          << "nu:  " << stage.nu()   << ", "
          << "ndx2:" << stage.ndx2();
    }

    if (stage.numConstraints() > 0)
    {
      oss << ", ";
      oss << "nc: " << stage.numConstraints();
    }
    
    oss << " }";
    return oss;
  }

  template<typename Scalar>
  StageDataTpl<Scalar>::StageDataTpl(const StageModel& stage_model)
    : constraint_data(stage_model.numConstraints())
    , cost_data(std::move(stage_model.cost_.createData()))
  {
    const std::size_t nc = stage_model.numConstraints();
    for (std::size_t i = 0; i < nc; i++)
    {
      const auto& func = stage_model.constraints_manager[i]->func_;
      constraint_data[i] = std::move(func.createData());
    }
  }
} // namespace proxddp
