/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/centroidal-kinematics-fwd.hpp"

namespace aligator {
namespace dynamics {

template struct CentroidalKinematicsFwdDynamicsTpl<context::Scalar>;
template struct CentroidalKinematicsFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
