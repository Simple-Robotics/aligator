#include "aligator/modelling/dynamics/ode-abstract.hpp"

namespace aligator {
namespace dynamics {

template struct ODEAbstractTpl<context::Scalar>;
template struct ODEDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
