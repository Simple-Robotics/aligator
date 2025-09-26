/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/context.hpp"
#include "aligator/math.hpp"
#include "aligator/gar/fwd.hpp"
#include <optional>

namespace aligator {
namespace gar {

template <typename _Scalar> class RiccatiSolverBase {
public:
  using Scalar = _Scalar;
  using LqrKnot = LqrKnotTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);

  virtual bool backward(const Scalar mueq) = 0;

  virtual bool
  forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
          std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
          const std::optional<ConstVectorRef> &theta_ = std::nullopt) const = 0;

  virtual void cycleAppend(const LqrKnot &knot) = 0;

  /// For applicable solvers, updates the first feedback gain in-place to
  /// correspond to the first Riccati gain.
  virtual void collapseFeedback() {}
  virtual VectorRef getFeedforward(size_t) = 0;
  virtual RowMatrixRef getFeedback(size_t) = 0;

  virtual ~RiccatiSolverBase() = default;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class RiccatiSolverBase<context::Scalar>;
#endif
} // namespace gar
} // namespace aligator
