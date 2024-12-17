/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/gar/lqr-problem.hxx"

namespace aligator {
namespace gar {
template struct LqrKnotTpl<context::Scalar>;
template struct LQRProblemTpl<context::Scalar>;

static_assert(
    std::uses_allocator_v<LqrKnotTpl<context::Scalar>, polymorphic_allocator>,
    "");
static_assert(std::uses_allocator_v<LQRProblemTpl<context::Scalar>,
                                    polymorphic_allocator>,
              "");

} // namespace gar
} // namespace aligator
