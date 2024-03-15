#pragma once

#include "aligator/context.hpp"
#include "aligator/core/common-model-abstract.hpp"

namespace aligator {

extern template struct CommonModelTpl<context::Scalar>;
extern template struct CommonModelDataTpl<context::Scalar>;
extern template class CommonModelBuilderTpl<context::Scalar>;

} // namespace aligator
