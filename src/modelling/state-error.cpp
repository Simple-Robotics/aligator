#include "proxddp/modelling/state-error.hpp"

namespace proxddp {

template struct StateErrorResidualTpl<context::Scalar>;
template struct ControlErrorResidualTpl<context::Scalar>;

} // namespace proxddp