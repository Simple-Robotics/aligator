#include "proxddp/modelling/state-error.hpp"

namespace aligator {

template struct StateErrorResidualTpl<context::Scalar>;
template struct ControlErrorResidualTpl<context::Scalar>;

} // namespace aligator
