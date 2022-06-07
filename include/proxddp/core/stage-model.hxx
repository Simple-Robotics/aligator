#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
  
  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numPrimal() const {
    return this->nu() + this->ndx2();
  }

  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numDual() const {
    return constraints_manager.totalDim();
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
          << "nu:  " << stage.nu() << ", "
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
    , dyn_data(constraint_data[0])
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
