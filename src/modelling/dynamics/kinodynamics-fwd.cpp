/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"

namespace aligator {
namespace dynamics {

template struct KinodynamicsFwdDynamicsTpl<context::Scalar>;
template struct KinodynamicsFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
