/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/integrator-rk2.hxx"

namespace aligator::dynamics {

template struct IntegratorRK2Tpl<context::Scalar>;
template struct IntegratorRK2DataTpl<context::Scalar>;

} // namespace aligator::dynamics
