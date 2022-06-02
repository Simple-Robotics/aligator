#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
  
  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numPrimal() const {
    return this->nu() + this->ndx2();
  }

  template<typename Scalar>
  inline int StageModelTpl<Scalar>::numDual() const {
    return constraints_.totalDim();
  }

} // namespace proxddp
