#pragma once

#include "riccati-base.hpp"
#include "riccati-impl.hpp"

namespace aligator {
namespace gar {

/// @brief A Riccati-like solver for the proximal LQ subproblem in ProxDDP.
template <typename _Scalar>
class ProximalRiccatiSolver : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using Base::datas;

  using Impl = ProximalRiccatiKernel<Scalar>;
  using StageFactorType = StageFactor<Scalar>;
  using value_t = typename StageFactorType::value_t;
  using kkt0_t = typename Impl::kkt0_t;
  using KnotType = LQRKnotTpl<Scalar>;

  explicit ProximalRiccatiSolver(const LQRProblemTpl<Scalar> &problem);

  /// Backward sweep.
  bool backward(const Scalar mudyn, const Scalar mueq);

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
               const std::optional<ConstVectorRef> &theta = std::nullopt) const;

  kkt0_t kkt0;     //< initial stage KKT system
  VectorXs thGrad; //< optimal value gradient wrt parameter
  MatrixXs thHess; //< optimal value Hessian wrt parameter

protected:
  void initialize();
  const LQRProblemTpl<Scalar> *problem_;
};

} // namespace gar
} // namespace aligator

#include "riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif
