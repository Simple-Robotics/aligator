#pragma once

#include "aligator/modelling/dynamics/fwd.hpp"
#include "aligator/context.hpp"

namespace aligator {
namespace context {

using ContinuousDynamicsAbstract =
    dynamics::ContinuousDynamicsAbstractTpl<Scalar>;

using ContinuousDynamicsData = dynamics::ContinuousDynamicsDataTpl<Scalar>;

using ODEAbstract = dynamics::ODEAbstractTpl<Scalar>;

using ODEData = dynamics::ODEDataTpl<Scalar>;

using IntegratorAbstract = dynamics::IntegratorAbstractTpl<Scalar>;

using IntegratorData = dynamics::IntegratorDataTpl<Scalar>;

} // namespace context
} // namespace aligator
