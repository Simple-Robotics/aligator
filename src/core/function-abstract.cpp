#include "aligator/core/function-abstract.hxx"

namespace aligator {

template struct StageFunctionTpl<context::Scalar>;
template struct StageFunctionDataTpl<context::Scalar>;
template std::ostream &
operator<<(std::ostream &oss,
           const StageFunctionDataTpl<context::Scalar> &self);

} // namespace aligator
