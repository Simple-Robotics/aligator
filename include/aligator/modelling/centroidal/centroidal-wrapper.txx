#pragma once

#include "aligator/context.hpp"
#include "aligator/modelling/centroidal/centroidal-wrapper.hpp"

namespace aligator {

extern template struct CentroidalWrapperResidualTpl<context::Scalar>;
extern template struct CentroidalWrapperDataTpl<context::Scalar>;

} // namespace aligator
