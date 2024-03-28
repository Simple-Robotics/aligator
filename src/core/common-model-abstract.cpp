#include "aligator/core/common-model-abstract.hpp"

namespace aligator {

template struct CommonModelTpl<context::Scalar>;
template struct CommonModelDataTpl<context::Scalar>;
template class CommonModelBuilderTpl<context::Scalar>;

} // namespace aligator
