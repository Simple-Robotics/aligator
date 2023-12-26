#pragma once

#include "./riccati-impl.hpp"
#include "aligator/threads.hpp"

namespace aligator {
namespace gar {

template <typename _Scalar> class ParallelRiccatiSolver {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class ParallelRiccatiSolver<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
