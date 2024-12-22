/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/gar/parallel-solver.hxx"

namespace aligator {
namespace gar {
#ifdef ALIGATOR_MULTITHREADING
template class ParallelRiccatiSolver<context::Scalar>;
#endif
} // namespace gar
} // namespace aligator
