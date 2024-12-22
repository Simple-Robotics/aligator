#pragma once

#include "riccati-base.hpp"

#include "aligator/context.hpp"

namespace aligator::gar {
extern template class RiccatiSolverBase<context::Scalar>;

}
