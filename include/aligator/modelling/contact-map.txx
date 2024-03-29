/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/contact-map.hpp"

namespace aligator {

extern template struct ContactMapTpl<context::Scalar>;

} // namespace aligator
