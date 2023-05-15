#include "proxddp/modelling/function-xpr-slice.hpp"

namespace proxddp {

template struct FunctionSliceXprTpl<context::Scalar, context::StageFunction>;
template struct FunctionSliceXprTpl<context::Scalar, context::UnaryFunction>;

} // namespace proxddp
