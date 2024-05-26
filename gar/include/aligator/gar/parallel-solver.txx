/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/gar/parallel-solver.hpp"

namespace aligator::gar {
#ifdef ALIGATOR_MULTITHREADING
extern template class ParallelRiccatiSolver<context::Scalar>;
#endif
} // namespace aligator::gar
