#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
  
  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numPrimal() const {
    return this->nu() + this->ndx2();
  }

  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numDual() const {
    int ret = dyn_model_.nr;
    for (std::size_t i = 0; i < numConstraints(); i++)
    {
      const StageFunctionTpl<Scalar>& func = constraints_[i]->func_;
      ret += func.nr;
    }
    return ret;
  }

} // namespace proxddp
