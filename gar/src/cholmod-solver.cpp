/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#include "aligator/gar/cholmod-solver.hxx"

#ifdef ALIGATOR_WITH_CHOLMOD
namespace aligator::gar {
template class CholmodLqSolver<context::Scalar>;
} // namespace aligator::gar
#endif
