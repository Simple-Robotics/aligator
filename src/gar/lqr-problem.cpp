/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/gar/lqr-problem.hxx"

namespace aligator {
namespace gar {
template struct LqrKnotTpl<context::Scalar>;
template struct LqrProblemTpl<context::Scalar>;

static_assert(
    std::uses_allocator_v<LqrKnotTpl<context::Scalar>, polymorphic_allocator>,
    "");
static_assert(std::uses_allocator_v<LqrProblemTpl<context::Scalar>,
                                    polymorphic_allocator>,
              "");

} // namespace gar
} // namespace aligator
