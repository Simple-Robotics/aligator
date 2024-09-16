#include "aligator/compat/crocoddyl/state-wrap.hpp"
#include "aligator/compat/crocoddyl/cost-wrap.hpp"

namespace aligator::compat::croc {

template struct CrocCostModelWrapperTpl<context::Scalar>;
template struct CrocCostDataWrapperTpl<context::Scalar>;

} // namespace aligator::compat::croc
