#include "aligator/modelling/function-xpr-slice.hpp"

namespace aligator {

template struct FunctionSliceXprTpl<context::Scalar, context::StageFunction>;
template struct FunctionSliceXprTpl<context::Scalar, context::UnaryFunction>;

} // namespace aligator
