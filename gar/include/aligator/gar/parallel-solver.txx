/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "parallel-solver.hpp"

namespace aligator::gar {
extern template class ParallelRiccatiSolver<context::Scalar>;
} // namespace aligator::gar
