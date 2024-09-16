#include "aligator/compat/crocoddyl/action-model-wrap.hxx"

namespace aligator::compat::croc {

template struct ActionModelWrapperTpl<context::Scalar>;
template struct ActionDataWrapperTpl<context::Scalar>;

} // namespace aligator::compat::croc
