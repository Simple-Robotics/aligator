#pragma once

#include "./riccati-impl.hpp"

namespace aligator {
namespace gar {

template <typename _Scalar> class ProximalRiccatiSolver {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using Impl = ProximalRiccatiImpl<Scalar>;
  using stage_factor_t = typename Impl::stage_factor_t;
  using value_t = typename Impl::value_t;
  using kkt0_t = typename Impl::kkt0_t;
  using knot_t = LQRKnotTpl<Scalar>;

  explicit ProximalRiccatiSolver(const LQRProblemTpl<Scalar> &problem)
      : datas(), kkt0(problem.stages[0].nx, problem.nc0(), problem.ntheta()),
        thGrad(problem.ntheta()), thHess(problem.ntheta(), problem.ntheta()),
        problem(problem) {
    initialize();
  }

  ProximalRiccatiSolver(LQRProblemTpl<Scalar> &&problem) = delete;

  /// Backward sweep.
  bool backward(const Scalar mudyn, const Scalar mueq) {
    ALIGATOR_NOMALLOC_BEGIN;
    bool ret = Impl::backwardImpl(problem.stages, mudyn, mueq, datas);

    stage_factor_t &d0 = datas[0];
    value_t &vinit = d0.vm;
    vinit.Vxx = vinit.Pmat;
    vinit.vx = vinit.pvec;
    // initial stage
    {
      kkt0.mat(0, 0) = vinit.Vxx;
      kkt0.mat(1, 0) = problem.G0;
      kkt0.mat(0, 1) = problem.G0.transpose();
      kkt0.mat(1, 1).diagonal().setConstant(-mudyn);
      kkt0.chol.compute(kkt0.mat.matrix());

      kkt0.ff.blockSegment(0) = -vinit.vx;
      kkt0.ff.blockSegment(1) = -problem.g0;
      kkt0.chol.solveInPlace(kkt0.ff.matrix());
      kkt0.fth.blockRow(0) = -vinit.Vxt;
      kkt0.fth.blockRow(1).setZero();
      kkt0.chol.solveInPlace(kkt0.fth.matrix());

      thGrad.noalias() =
          vinit.vt + vinit.Vxt.transpose() * kkt0.ff.blockSegment(0);
      thHess.noalias() =
          vinit.Vtt + vinit.Vxt.transpose() * kkt0.fth.blockRow(0);
    }
    ALIGATOR_NOMALLOC_END;
    return ret;
  }

  bool
  forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
          std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
          const boost::optional<ConstVectorRef> &theta_ = boost::none) const {

    // solve initial stage
    Impl::computeInitial(xs[0], lbdas[0], kkt0, theta_);

    return Impl::forwardImpl(problem.stages, datas, xs, us, vs, lbdas, theta_);
  }

  std::vector<stage_factor_t> datas;
  kkt0_t kkt0;

  VectorXs thGrad; //< optimal value gradient wrt parameter
  MatrixXs thHess; //< optimal value Hessian wrt parameter

protected:
  void initialize() {
    auto N = uint(problem.horizon());
    auto &knots = problem.stages;
    datas.reserve(N + 1);
    for (uint t = 0; t <= N; t++) {
      const knot_t &knot = knots[t];
      datas.emplace_back(knot.nx, knot.nu, knot.nc, knot.nth);
    }
    thGrad.setZero();
    thHess.setZero();
    kkt0.mat.setZero();
  }
  const LQRProblemTpl<Scalar> &problem;
};

} // namespace gar
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif
