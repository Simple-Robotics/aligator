#pragma once

#include "riccati-impl.hpp"
#include <tbb/cache_aligned_allocator.h>

namespace aligator {
namespace gar {

template <typename _Scalar> class RiccatiSolverBase {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageFactorType = StageFactor<Scalar>;
  using StageFactorVec =
      std::vector<StageFactorType,
                  tbb::cache_aligned_allocator<StageFactorType>>;
  StageFactorVec datas;

  virtual bool backward(const Scalar mudyn, const Scalar mueq) = 0;

  virtual bool
  forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
          std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
          const std::optional<ConstVectorRef> &theta_ = std::nullopt) const = 0;

  virtual ~RiccatiSolverBase() = default;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class RiccatiSolverBase<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator
