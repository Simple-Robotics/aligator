#include "proxddp/modelling/dynamics/ode-abstract.hpp"

namespace proxddp {
namespace dynamics {

template struct ODEAbstractTpl<context::Scalar>;
template struct ODEDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace proxddp
