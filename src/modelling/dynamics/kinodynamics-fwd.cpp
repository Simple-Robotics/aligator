/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/dynamics/kinodynamics-fwd.hxx"

namespace aligator {
namespace dynamics {

template struct KinodynamicsFwdDynamicsTpl<context::Scalar>;
template struct KinodynamicsFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
#endif
