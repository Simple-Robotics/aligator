/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/dynamics.hpp"

namespace aligator {

template <typename Scalar>
DynamicsModelTpl<Scalar>::DynamicsModelTpl(ManifoldPtr space, const int nu)
    : Base(space->ndx(), nu, space->ndx(), space->ndx()), space_(space),
      space_next_(space) {}

template <typename Scalar>
DynamicsModelTpl<Scalar>::DynamicsModelTpl(ManifoldPtr space, const int nu,
                                           ManifoldPtr space2)
    : Base(space->ndx(), nu, space2->ndx(), space2->ndx()), space_(space),
      space_next_(space2) {}

} // namespace aligator
