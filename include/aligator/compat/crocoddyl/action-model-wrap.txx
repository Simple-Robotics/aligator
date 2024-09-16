#pragma once

#include "aligator/compat/crocoddyl/action-model-wrap.hpp"

namespace aligator::compat::croc {

extern template struct ActionModelWrapperTpl<context::Scalar>;
extern template struct ActionDataWrapperTpl<context::Scalar>;

} // namespace aligator::compat::croc
