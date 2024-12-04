/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/kinodynamics-fwd.hpp"
#ifdef PROXSUITE_NLP_WITH_PINOCCHIO

namespace aligator {
namespace dynamics {

template struct KinodynamicsFwdDynamicsTpl<context::Scalar>;
template struct KinodynamicsFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
#endif
